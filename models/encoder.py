from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # input (channels, width, height) is    (3,  128, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),    # (8,  128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (8,  64,  64)

            nn.Conv2d(8, 16, 3, padding=1),   # (16, 64,  64)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (16, 32,  32)

            nn.Conv2d(16, 32, 3, padding=1),  # (32, 32,  32)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (32, 16,  16)
        )

    def forward(self, x):
        return self.encoder(x)
