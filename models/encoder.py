from torch import nn
import torch.nn.functional as f


class LowerEncoder(nn.Module):
    """
    Domain-specific encoder stage for features of low level of abstraction.
    """

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 7, padding=3)
        self.conv1_2 = nn.Conv2d(64, 64, 4, padding=1, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)

    def forward(self, x):
        x = f.relu(self.conv1_1(x))
        x = f.relu(self.conv1_2(x))

        x = f.relu(self.conv2_1(x))
        x = f.relu(self.conv2_2(x))

        x = f.relu(self.conv3_1(x))
        x = f.relu(self.conv3_2(x))
        x = f.relu(self.conv3_3(x))

        x = f.relu(self.conv4_1(x))

        return x


class UpperEncoder(nn.Module):
    """
    Domain-invariant encoder stage for features of high level of abstraction. Is supposed
    to be shared by multiple AutoEncoder
    """

    def __init__(self):
        super().__init__()

        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU()

    def forward(self, x):
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))

        return x
