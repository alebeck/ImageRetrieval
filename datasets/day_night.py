import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class DayNightDataset(Dataset):

    def __init__(self, path_day, path_night):

        self.images_day, self.images_night = [], []
        to_tensor = ToTensor()

        for path in os.listdir(path_day):
            with open(path, 'rb') as file:
                img = Image.open(file)
                self.images_day.append(to_tensor(img))

        for path in os.listdir(path_night):
            with open(path, 'rb') as file:
                img = Image.open(file)
                self.images_night.append(to_tensor(img))

    def __len__(self):
        return min(len(self.images_day), len(self.images_night))

    def __getitem__(self, index):
        return self.images_day[index], self.images_night[index]
