import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from dofa_wave_layer import Dynamic_MLP_OFA

# Checkpoints auto-download from Hugging Face via torch.hub's own
# caching (no manual file hunting needed, unlike SkySense). CC-BY-4.0
# licensed -- see https://huggingface.co/earthflow/DOFA.
DOFA_CHECKPOINT_URLS = {
    "vit_base_patch16": "https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth",
    "vit_large_patch16": "https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_large_e100.pth"
}

# (embed_dim, depth, num_heads, out_indices) per variant. out_indices
# are the transformer block indices DOFA's own segmentation reference
# implementation (downstream_tasks/geobench_segmentation/model.py)
# extracts features from -- 4 different depths, but note these are
# all at the SAME spatial resolution (H/patch_size), since a plain ViT
# has no hierarchical downsampling like Swin. See PseudoPyramidNeck
# below for how this gets turned into an actual multi-resolution
# pyramid for the decoder.
DOFA_VARIANTS = {
    "vit_base_patch16": {"embed_dim": 768, "depth": 12, "num_heads": 12, "out_indices": [3, 5, 7, 11]},
    "vit_large_patch16": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "out_indices": [7, 11, 15, 23]}
}


class DOFABackbone(nn.Module):
    """DOFA's ViT backbone with wavelength-conditioned dynamic patch
    embedding (adapted from downstream_tasks/geobench_segmentation/
    model.py in the DOFA repo). Returns a list of 4 feature maps, one
    per out_indices entry, each shaped (B, embed_dim, H/16, W/16) --
    same spatial resolution, different transformer depths."""

    def __init__(self, variant="vit_base_patch16", patch_size=16, img_size=512):
        super().__init__()

        config = DOFA_VARIANTS[variant]

        self.out_indices = config["out_indices"]
        self.img_size = img_size
        self.patch_size = patch_size
        embed_dim = config["embed_dim"]

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, config["num_heads"], mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(config["depth"])
        ])

        self.embed_dim = embed_dim

    def forward(self, x, wavelengths):
        hw = self.img_size // self.patch_size

        x, _ = self.patch_embed(x, wavelengths)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in self.out_indices:
                out = x[:, 1:]
                batch, _, channels = out.shape
                out = out.reshape(batch, hw, hw, channels).permute(0, 3, 1, 2).contiguous()
                outputs.append(out)

        return outputs


class PseudoPyramidNeck(nn.Module):
    """Converts DOFA's 4 same-resolution, different-depth feature maps
    into an actual multi-resolution pyramid (mirrors mmseg's
    Feature2Pyramid(rescales=[4,2,1,0.5]), reimplemented here in plain
    PyTorch to avoid an mmseg/mmcv dependency): the shallowest output
    gets upsampled 4x, the next 2x, the third stays as-is, and the
    deepest gets downsampled 2x. This gives a normal FPN-style pyramid
    (H/4, H/8, H/16, H/32) that a conventional UNet decoder can consume
    with skip connections, the same way it would consume a hierarchical
    CNN/Swin encoder's actual stage outputs."""

    def __init__(self, embed_dim):
        super().__init__()

        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        )
        self.upsample_2x = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.identity = nn.Identity()
        self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, features):
        shallow, mid_shallow, mid_deep, deep = features

        level_h4 = self.upsample_4x(shallow)
        level_h8 = self.upsample_2x(mid_shallow)
        level_h16 = self.identity(mid_deep)
        level_h32 = self.downsample_2x(deep)

        return level_h4, level_h8, level_h16, level_h32


class DecoderBlock(nn.Module):
    """One up-sampling decoder stage: upsample the deeper feature map,
    concatenate with the matching pyramid level, then refine with two
    conv layers. Same design as skysense_model.py's DecoderBlock."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


def _interpolate_pos_embed(state_dict, target_num_patches, embed_dim):
    """Bicubically interpolates DOFA's positional embedding from
    whatever grid it was pretrained at (224x224 images -> 14x14 = 196
    patches) to the grid this project's tiles actually use (512x512
    images -> 32x32 = 1024 patches). Without this, loading the
    checkpoint into a differently-sized model fails with a pos_embed
    shape mismatch -- this is standard practice when fine-tuning a ViT
    at a different resolution than it was pretrained at (adapted from
    DOFA's own pretraining/util/pos_embed.py::interpolate_pos_embed).
    The cls token's own position embedding entry is left untouched;
    only the per-patch entries are resized."""

    if "pos_embed" not in state_dict:
        return state_dict

    pos_embed_checkpoint = state_dict["pos_embed"]
    num_extra_tokens = 1  # cls token

    orig_num_patches = pos_embed_checkpoint.shape[1] - num_extra_tokens
    orig_size = int(orig_num_patches ** 0.5)
    new_size = int(target_num_patches ** 0.5)

    if orig_size == new_size:
        return state_dict

    print(f"  Interpolating DOFA pos_embed: {orig_size}x{orig_size} -> {new_size}x{new_size} patches")

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    state_dict["pos_embed"] = torch.cat([extra_tokens, pos_tokens], dim=1)

    return state_dict


def _download_dofa_checkpoint(variant, cache_dir):
    """Downloads (or reuses a cached copy of) the pretrained DOFA
    checkpoint via torch.hub's own caching -- no manual file placement
    needed, unlike SkySense."""

    os.makedirs(cache_dir, exist_ok=True)

    url = DOFA_CHECKPOINT_URLS[variant]

    state_dict = torch.hub.load_state_dict_from_url(
        url, model_dir=cache_dir, map_location="cpu"
    )

    return state_dict


class DOFAUNet(nn.Module):
    """Drop-in replacement for model.py's UNet / skysense_model.py's
    SkySenseUNet -- same forward(x) -> logits signature -- but the
    encoder is DOFA's pretrained, wavelength-conditioned ViT backbone,
    which (unlike SkySense) can accept ANY number of input channels.

    `wavelengths` is a list of per-channel center wavelengths in
    micrometers, in the SAME channel order as the input tensor. Real
    optical wavelengths for RGB (e.g. [0.66, 0.56, 0.48]); for
    DTM/hillshade/slope there is no true wavelength, so these are
    placeholder pseudo-wavelengths only -- see config.py's
    DOFA_WAVELENGTHS for the exact values and a fuller explanation of
    this caveat.

    freeze_backbone=True (the default) freezes every backbone
    parameter so only the new neck+decoder train -- the recommended
    first stage of a staged fine-tune. Call unfreeze_backbone_blocks()
    later to unfreeze the last N transformer blocks at a lower
    learning rate."""

    DECODER_CHANNELS = [512, 256, 128, 64]

    def __init__(
            self,
            wavelengths,
            checkpoint_dir,
            variant="vit_base_patch16",
            out_channels=1,
            freeze_backbone=True,
            input_size=512
    ):
        super().__init__()

        self.register_buffer("wavelengths", torch.tensor(wavelengths, dtype=torch.float32))

        self.backbone = DOFABackbone(variant=variant, img_size=input_size)

        state_dict = _download_dofa_checkpoint(variant, checkpoint_dir)
        state_dict = _interpolate_pos_embed(state_dict, self.backbone.num_patches, self.backbone.embed_dim)
        load_result = self.backbone.load_state_dict(state_dict, strict=False)

        print(f"DOFA backbone ({variant}) loaded.")
        print("  Missing keys:", load_result.missing_keys)
        print("  Unexpected keys:", load_result.unexpected_keys)

        embed_dim = self.backbone.embed_dim

        self.neck = PseudoPyramidNeck(embed_dim)

        d0, d1, d2, d3 = self.DECODER_CHANNELS

        self.dec1 = DecoderBlock(embed_dim, embed_dim, d0)
        self.dec2 = DecoderBlock(d0, embed_dim, d1)
        self.dec3 = DecoderBlock(d1, embed_dim, d2)

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(d2, d3, kernel_size=2, stride=2),
            nn.BatchNorm2d(d3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d3, d3, kernel_size=2, stride=2),
            nn.BatchNorm2d(d3),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(d3, out_channels, kernel_size=1)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        print("DOFA backbone frozen (neck+decoder-only training).")

    def unfreeze_backbone_blocks(self, num_blocks=2):
        """Unfreezes the last `num_blocks` transformer blocks, for the
        second phase of a staged fine-tune. Pair with a much lower
        learning rate for these params -- see
        differential_param_groups()."""

        blocks_to_unfreeze = list(self.backbone.blocks)[-num_blocks:]

        for block in blocks_to_unfreeze:
            for parameter in block.parameters():
                parameter.requires_grad = True

        print(f"Unfroze the last {num_blocks} DOFA transformer block(s).")

    def differential_param_groups(self, decoder_lr, backbone_lr):
        """Optimizer param groups: the neck+decoder always train at
        decoder_lr; any currently-unfrozen backbone parameters train
        at the (typically much lower) backbone_lr."""

        decoder_modules = [self.neck, self.dec1, self.dec2, self.dec3, self.final_upsample, self.output]
        decoder_params = [parameter for module in decoder_modules for parameter in module.parameters()]

        backbone_params = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

        groups = [{"params": decoder_params, "lr": decoder_lr}]

        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})

        return groups

    def forward(self, x):
        features = self.backbone(x, self.wavelengths)

        level_h4, level_h8, level_h16, level_h32 = self.neck(features)

        x = self.dec1(level_h32, level_h16)
        x = self.dec2(x, level_h8)
        x = self.dec3(x, level_h4)
        x = self.final_upsample(x)

        return self.output(x)
