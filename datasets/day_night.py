import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image


class DayNightDataset(Dataset):

    def __init__(self, paths_day, paths_night, img_size=128):
        self.images_day, self.images_night = [], []

        transform = Compose([
            Resize(img_size),
            ToTensor()
        ])

        print("Loading data...")

        for path in paths_day:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    self.images_day.append(transform(img))

        for path in paths_night:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    self.images_night.append(transform(img))

    def __len__(self):
        return min(len(self.images_day), len(self.images_night))

    def __getitem__(self, index):
        return self.images_day[index], self.images_night[index]
