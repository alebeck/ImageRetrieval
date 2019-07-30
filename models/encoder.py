from torch import nn
import torch.nn.functional as f


class LowerEncoder(nn.Module):
    """
    Domain-specific encoder stage for features of low level of abstraction.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=3)

        # downsampling layers
        self.conv2_1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)

        # residual blocks
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        x = f.relu(self.conv1(x))

        x = f.relu(self.conv2_1(x))
        x = f.relu(self.conv2_2(x))

        res = x
        x = f.relu(self.conv3_1(x))
        x = self.conv3_2(x)
        x += res

        res = x
        x = f.relu(self.conv4_1(x))
        x = self.conv4_2(x)
        x += res

        return x


class UpperEncoder(nn.Module):
    """
    Domain-invariant encoder stage for features of high level of abstraction. Is supposed
    to be shared by multiple AutoEncoder
    """

    def __init__(self):
        super().__init__()

        self.conv5_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        res = x
        x = f.relu(self.conv5_1(x))
        x = self.conv5_2(x)
        x += res

        res = x
        x = f.relu(self.conv6_1(x))
        x = self.conv6_2(x)
        x += res

        return x
