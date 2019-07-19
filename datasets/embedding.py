import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from models.abstract import CustomModule, EmbeddingGenerator
from utils.functions import unit_normalize


class EmbeddingDataset(Dataset):

    def __init__(self, model_class, model_args, weights_path, paths_day, paths_night, layers, transform=None, unit_norm=True, count=400):
        use_cuda = torch.cuda.is_available()

        model: (CustomModule, EmbeddingGenerator) = model_class(**model_args)
        if weights_path is not None:
            if use_cuda:
                model.load_state_dict(torch.load(weights_path)['model'])
            else:
                model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])

        if use_cuda:
            model.cuda()

        model.eval()

        if transform is None:
            transform = ToTensor()

        self.day_files, self.night_files = [], []
        for path in paths_day:
            for filename in sorted(os.listdir(path)):
                if not filename.startswith('.'):
                    self.day_files.append(os.path.join(path, filename))

        for path in paths_night:
            for filename in sorted(os.listdir(path)):
                if not filename.startswith('.'):
                    self.night_files.append(os.path.join(path, filename))

        idx = np.random.choice(np.arange(min(len(self.day_files), len(self.night_files))), count, replace=False)

        imgs_day, imgs_night = [], []
        with torch.no_grad():
            for i in idx:
                with open(self.day_files[i], 'rb') as file:
                    img = transform(Image.open(file)).unsqueeze(0)
                    imgs_day.append(img)

                with open(self.night_files[i], 'rb') as file:
                    img = transform(Image.open(file)).unsqueeze(0)
                    imgs_night.append(img)

        imgs_day, imgs_night = torch.cat(imgs_day), torch.cat(imgs_night)
        if use_cuda:
            imgs_day, imgs_night = imgs_day.cuda(), imgs_night.cuda()

        embeddings_day = model.get_day_embeddings(imgs_day, layers)
        embeddings_night = model.get_night_embeddings(imgs_night, layers)

        self.embeddings_day, self.embeddings_night = [], []

        for i in range(count):
            new_emb = {}
            for layer in layers:
                new_emb[layer] = unit_normalize(embeddings_day[layer][i]) if unit_norm else embeddings_day[layer][i]
            self.embeddings_day.append(new_emb)

        for i in range(count):
            new_emb = {}
            for layer in layers:
                new_emb[layer] = unit_normalize(embeddings_night[layer][i]) if unit_norm else embeddings_night[layer][i]
            self.embeddings_night.append(new_emb)

        print('Done')

    def __len__(self):
        return min(len(self.embeddings_day), len(self.embeddings_night))

    def __getitem__(self, index):
        return self.embeddings_day[index], self.embeddings_night[index]