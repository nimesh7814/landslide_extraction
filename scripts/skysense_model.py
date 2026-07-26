import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Expects a local clone of https://github.com/Jack-bo1220/SkySense at
# this path (sibling to this file):
#   git clone https://github.com/Jack-bo1220/SkySense.git skysense_repo
#
# That repo's backbone code depends on mmcv-full==1.7.1 and
# mmcls==0.25.0 (see its README for the full environment setup). This
# module only imports from it inside _load_backbone(), so simply
# importing skysense_model.py (e.g. from 2_train_model.py) does not
# require that environment -- only actually constructing a
# SkySenseUNet does.
SKYSENSE_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skysense_repo")

if SKYSENSE_REPO_DIR not in sys.path:
    sys.path.insert(0, SKYSENSE_REPO_DIR)


def _load_backbone(checkpoint_path, out_indices=(0, 1, 2, 3)):
    """Builds the SkySense Swin Transformer V2 - Huge backbone and loads
    its pretrained weights (the 'hr' / high-resolution RGB checkpoint).
    Requires mmcv-full==1.7.1 and mmcls==0.25.0 to be installed -- see
    SKYSENSE_REPO_DIR's own README."""

    try:
        from models.swin_transformer_v2 import SwinTransformerV2
    except ImportError as error:
        raise ImportError(
            "Could not import SwinTransformerV2 from the vendored SkySense "
            f"repo at {SKYSENSE_REPO_DIR}. Make sure you've cloned it there "
            "and installed mmcv-full==1.7.1 and mmcls==0.25.0 in this "
            "environment (see the SkySense README)."
        ) from error

    # out_indices=(0,1,2,3) returns all 4 hierarchical stage feature
    # maps (SkySense's own default only returns the last stage, which
    # is fine for classification but not for a UNet-style decoder).
    # pad_small_map=True lets the last stage's small feature map avoid
    # errors if it ends up smaller than window_size.
    backbone = SwinTransformerV2(
        arch="huge",
        out_indices=out_indices,
        pad_small_map=True
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SkySense checkpoint not found: {checkpoint_path}. Download "
            "skysense_model_backbone_hr.pth from the SkySense repo's README "
            "(linked Notion checkpoints page) first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = {
        key.replace("backbone.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("backbone.")
    }

    load_result = backbone.load_state_dict(checkpoint, strict=False)

    print("SkySense backbone loaded from:", checkpoint_path)
    print("  Missing keys:", load_result.missing_keys)
    print("  Unexpected keys:", load_result.unexpected_keys)

    return backbone


class DecoderBlock(nn.Module):
    """One up-sampling decoder stage: upsample the deeper feature map,
    concatenate with the matching encoder skip connection, then refine
    with two conv layers. Same skip-connection idea as model.py's
    UNet, just sized to whatever channel counts the Swin backbone
    stages actually have (discovered at runtime, not hardcoded)."""

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


class SkySenseUNet(nn.Module):
    """Drop-in replacement for model.py's UNet -- same forward(x) ->
    logits signature -- but the encoder is a SkySense-pretrained Swin
    Transformer V2 - Huge backbone instead of a from-scratch conv
    encoder.

    Only supports 3-channel RGB input: this is the only SkySense
    pretrained variant that matches this project's data (see
    01_ortho_dataset). The DTM/hillshade/slope dataset variants have
    no matching pretrained SkySense modality.

    freeze_backbone=True (the default) freezes every backbone
    parameter so only the new decoder trains -- the recommended first
    stage of a staged fine-tune. Call unfreeze_backbone_stages()
    later to unfreeze the last N stages at a lower learning rate."""

    DECODER_CHANNELS = [512, 256, 128, 64]

    # SkySense's own iSAID segmentation config (segmentation/configs/
    # _base_/datasets/isaid.py) normalizes with mean=[123.675, 116.28,
    # 103.53], std=[58.395, 57.12, 57.375] on 0-255 pixel values --
    # standard ImageNet stats. This project's tiles are already scaled
    # to 0-1 (not 0-255) by 1_create_train_dataset.py, so the
    # equivalent here is ImageNet's usual 0-1-scale mean/std. Applied
    # here, inline, rather than by re-tiling -- the .npy tiles stay a
    # generic 0-1 RGB representation usable by any model.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
            self,
            checkpoint_path,
            out_channels=1,
            freeze_backbone=True,
            input_size=512
    ):
        super().__init__()

        self.register_buffer("input_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        self.backbone = _load_backbone(checkpoint_path, out_indices=(0, 1, 2, 3))

        # Discover each stage's actual output channel count with a
        # small dummy forward pass, rather than hardcoding Swin's
        # channel-doubling schedule -- keeps this robust to any
        # architecture detail this wrapper doesn't need to know about.
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy = (dummy - self.input_mean) / self.input_std
            stage_outputs = self.backbone(dummy)

        stage_channels = [stage_output.shape[1] for stage_output in stage_outputs]
        print("SkySense backbone stage channels:", stage_channels)

        c0, c1, c2, c3 = stage_channels
        d0, d1, d2, d3 = self.DECODER_CHANNELS

        # Decode from the deepest stage (c3) back up to the shallowest
        # (c0), then two more upsamples to undo the patch embedding's
        # own 4x downsampling and reach full input resolution.
        self.dec1 = DecoderBlock(c3, c2, d0)
        self.dec2 = DecoderBlock(d0, c1, d1)
        self.dec3 = DecoderBlock(d1, c0, d2)

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

        print("SkySense backbone frozen (decoder-only training).")

    def unfreeze_backbone_stages(self, num_stages=1):
        """Unfreezes the last `num_stages` backbone stages (out of 4),
        for the second phase of a staged fine-tune. Pair with a much
        lower learning rate for these params -- see
        differential_param_groups()."""

        stages_to_unfreeze = list(self.backbone.stages)[-num_stages:]

        for stage in stages_to_unfreeze:
            for parameter in stage.parameters():
                parameter.requires_grad = True

        print(f"Unfroze the last {num_stages} SkySense backbone stage(s).")

    def differential_param_groups(self, decoder_lr, backbone_lr):
        """Optimizer param groups: the decoder always trains at
        decoder_lr; any currently-unfrozen backbone parameters train
        at the (typically much lower) backbone_lr. Pass straight to
        torch.optim.Adam(model.differential_param_groups(...))."""

        decoder_modules = [self.dec1, self.dec2, self.dec3, self.final_upsample, self.output]
        decoder_params = [parameter for module in decoder_modules for parameter in module.parameters()]

        backbone_params = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

        groups = [{"params": decoder_params, "lr": decoder_lr}]

        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})

        return groups

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std

        stage0, stage1, stage2, stage3 = self.backbone(x)

        x = self.dec1(stage3, stage2)
        x = self.dec2(x, stage1)
        x = self.dec3(x, stage0)
        x = self.final_upsample(x)

        return self.output(x)
