import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image


class DayNightDataset(Dataset):

    def __init__(self, path_day, path_night, img_size=128):
        self.images_day, self.images_night = [], []

        transform = Compose([
            Resize(img_size),
            ToTensor()
        ])

        for filename in os.listdir(path_day):
            with open(os.path.join(path_day, filename), 'rb') as file:
                img = Image.open(file)
                self.images_day.append(transform(img))

        for filename in os.listdir(path_night):
            with open(os.path.join(path_night, filename), 'rb') as file:
                img = Image.open(file)
                self.images_night.append(transform(img))

    def __len__(self):
        return min(len(self.images_day), len(self.images_night))

    def __getitem__(self, index):
        return self.images_day[index], self.images_night[index]
