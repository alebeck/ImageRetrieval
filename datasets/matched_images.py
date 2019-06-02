import os
from random import randint

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image


class MatchedImagesDataset(Dataset):
    """
    Loads matched day/night images as triplets of anchor, positive and negative, where the anchor matches the positive,
    but not the negative image.
    """

    def __init__(self, paths_anchors: [str], paths_opposites: [str], img_size: int = 128):
        self.anchors, self.opposites = [], []

        transform = Compose([
            Resize(img_size),
            ToTensor()
        ])

        print("Loading data...")

        for path in paths_anchors:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    self.anchors.append(transform(img))

        for path in paths_opposites:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    self.opposites.append(transform(img))

    def __len__(self):
        return min(len(self.anchors), len(self.opposites))

    def __getitem__(self, index):
        negative_index = index
        while negative_index == index:
            negative_index = randint(0, len(self) - 1)

        anchor = self.anchors[index]
        positive = self.opposites[index]
        negative = self.opposites[negative_index]

        return anchor, positive, negative
