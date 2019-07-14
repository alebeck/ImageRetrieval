import torch
from torch import nn
import torch.nn.functional as f


class LowerDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        res = x
        x = f.relu(self.conv1_1(x))
        x = self.conv1_2(x)
        x += res

        res = x
        x = f.relu(self.conv2_1(x))
        x = self.conv2_2(x)
        x += res

        return x


class UpperDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        # residual blocks
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        # upsampling layers
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(256, 128, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv6 = nn.Conv2d(128, 64, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 3, 7, stride=1, padding=3)


    def forward(self, x):
        res = x
        x = f.relu(self.conv3_1(x))
        x = self.conv3_2(x)
        x += res

        res = x
        x = f.relu(self.conv4_1(x))
        x = self.conv4_2(x)
        x += res

        x = self.up1(x)
        x = f.relu(self.bn1(self.conv5(x))) # ln norm

        x = self.up2(x)
        x = f.relu(self.bn2(self.conv6(x))) # ln norm

        x = torch.tanh(self.conv7(x)) # padding reflect

        return x