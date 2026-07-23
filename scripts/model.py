import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 encoder_channels, decoder_channels,
                 bottleneck_channels):
        super().__init__()

        self.enc1 = double_conv(in_channels, encoder_channels[0])
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  double_conv(encoder_channels[0], encoder_channels[1]))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),
                                  double_conv(encoder_channels[1], encoder_channels[2]))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2),
                                  double_conv(encoder_channels[2], encoder_channels[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(encoder_channels[3], bottleneck_channels))

        self.up1 = up_conv(bottleneck_channels, bottleneck_channels)
        self.dec1 = double_conv(bottleneck_channels + encoder_channels[3], decoder_channels[0])

        self.up2 = up_conv(decoder_channels[0], decoder_channels[0])
        self.dec2 = double_conv(decoder_channels[0] + encoder_channels[2], decoder_channels[1])

        self.up3 = up_conv(decoder_channels[1], decoder_channels[1])
        self.dec3 = double_conv(decoder_channels[1] + encoder_channels[1], decoder_channels[2])

        self.up4 = up_conv(decoder_channels[2], decoder_channels[2])
        self.dec4 = double_conv(decoder_channels[2] + encoder_channels[0], decoder_channels[3])

        self.output = nn.Conv2d(decoder_channels[3], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        x = self.bottom(e4)

        x = self.up1(x)
        x = torch.cat([x, e4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec4(x)

        return self.output(x)
