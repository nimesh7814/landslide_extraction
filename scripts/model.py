import torch.nn as nn
import segmentation_models_pytorch as smp


class LandslideUNet(nn.Module):
    # UNet with a pretrained ResNet50 (ImageNet) encoder, built via segmentation_models_pytorch.

    def __init__(
            self,
            in_channels,
            out_channels=1,
            encoder_name="resnet50",
            encoder_weights="imagenet",
            freeze_encoder=True
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )

        print(f"UNet built (encoder={encoder_name}, weights={encoder_weights}, in_channels={in_channels}).")

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        # Freezes the pretrained encoder so only the decoder + segmentation head train.
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = False

        print("Encoder frozen (decoder+head-only training).")

    def unfreeze_encoder(self):
        # Unfreezes the encoder for the fine-tuning phase, once the decoder has warmed up.
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = True

        print("Encoder unfrozen (fine-tuning all encoder layers).")

    def differential_param_groups(self, decoder_lr, encoder_lr):
        # Decoder+head always train at decoder_lr; the encoder trains at the lower encoder_lr,
        # whether currently frozen or not, so unfreeze_encoder() needs no optimizer changes later.
        decoder_params = list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())
        encoder_params = list(self.model.encoder.parameters())

        return [
            {"params": decoder_params, "lr": decoder_lr},
            {"params": encoder_params, "lr": encoder_lr}
        ]

    def forward(self, x):
        return self.model(x)
