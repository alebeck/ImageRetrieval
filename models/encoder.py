from torch import nn
import torch.nn.functional as f


class LowerEncoder(nn.Module):
    """
    Domain-specific encoder stage for features of low level of abstraction.
    """

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        x = f.relu(self.conv1_1(x))
        x = f.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = f.relu(self.conv2_1(x))

        return x


class UpperEncoder(nn.Module):
    """
    Domain-invariant encoder stage for features of high level of abstraction. Is supposed
    to be shared by multiple AutoEncoder
    """

    def __init__(self):
        super().__init__()

        self.conv2_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = f.relu(self.conv2_2(x))
        x = f.relu(self.conv2_3(x))
        x = f.relu(self.conv2_4(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)

        return x
