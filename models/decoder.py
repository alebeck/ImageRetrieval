from torch import nn
import torch.nn.functional as f


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.convt1_1 = nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1, output_padding=0)
        self.convt1_2 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1, output_padding=0)
        self.convt1_3 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)

        self.convt2_1 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1, output_padding=0)
        self.convt2_2 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, output_padding=0)
        self.convt2_3 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, output_padding=0)
        self.convt2_4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.convt3_1 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0)
        self.convt3_2 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = f.relu(self.convt1_1(x))
        x = f.relu(self.convt1_2(x))
        x = f.relu(self.convt1_3(x))

        x = f.relu(self.convt2_1(x))
        x = f.relu(self.convt2_2(x))
        x = f.relu(self.convt2_3(x))
        x = f.relu(self.convt2_4(x))

        x = f.relu(self.convt3_1(x))
        x = f.relu(self.convt3_2(x))

        # limit to [0; 1]
        x = x.clamp(0., 1.)

        return x