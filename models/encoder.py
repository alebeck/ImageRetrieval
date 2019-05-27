from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # input (channels, width, height) is    (3,  128, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),    # (6,  128, 128)
            nn.ReLU(),
            nn.Conv2d(6, 10, 3, padding=1),   # (10, 128, 128)
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, padding=1),  # (16, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (16, 64,  64)

            nn.Conv2d(16, 24, 3, padding=1),  # (24, 64,  64)
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),  # (32, 64,  64)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (32, 32,  32)

            nn.Conv2d(32, 48, 3, padding=1),  # (48, 32,  32)
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1),  # (64, 32,  32)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # (64, 16,  16)
        )

    def forward(self, x):
        return self.encoder(x)
