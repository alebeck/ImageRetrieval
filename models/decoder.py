from torch import nn


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # input (channels, width, height) is                                         (64, 16,  16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 24, 3, stride=2, padding=1, output_padding=1),  # (24, 32,  32)
            nn.ReLU(),

            nn.ConvTranspose2d(24, 8, 3, stride=2, padding=1, output_padding=1),   # (8, 64,  64)
            nn.ReLU(),

            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),    # (3,  128, 128)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.decoder(x)
