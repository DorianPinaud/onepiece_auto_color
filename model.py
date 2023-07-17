import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        _common_args = {"kernel_size": 1, "stride": 1, "padding": 0, "bias": True}
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                **_common_args,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                **_common_args,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for conv in self.encoder:
            x = conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.decoder), 2):
            transpose_conv = self.decoder[index]
            x = transpose_conv(x)
            skip_conn = skip_connections[index // 2]
            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])
            x = torch.cat((x, skip_conn), dim=1)
            conv = self.decoder[index + 1]
            x = conv(x)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(1, 1)
    preds = model(x)
    print("preds.shape", preds.shape)
    print("x.shape", x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
