import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from models.abstract import CustomModule, EmbeddingGenerator
from utils.functions import unit_normalize


class EmbeddingDataset(Dataset):

    def __init__(self, model_class, model_args, weights_path, paths_day, paths_night, layers, transform=None, unit_norm=True, count=400, use_cuda=True):
        use_cuda = torch.cuda.is_available() and use_cuda

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

        self.day_files = []
        self.night_files = []

        for path in paths_day:
            for filename in sorted(os.listdir(path)):
                if not filename.startswith('.'):
                    self.day_files.append(os.path.join(path, filename))

        for path in paths_night:
            for filename in sorted(os.listdir(path)):
                if not filename.startswith('.'):
                    self.night_files.append(os.path.join(path, filename))

        idx = np.random.choice(np.arange(min(len(self.day_files), len(self.night_files))), count, replace=False)

        self.embeddings_day, self.embeddings_night = [], []

        with torch.no_grad():
            for count, i in enumerate(idx):

                print(f'\r{count}/{len(idx)}', end='', flush=True)

                with open(self.day_files[i], 'rb') as file:
                    img = transform(Image.open(file)).unsqueeze(0)
                    if use_cuda:
                        img = img.cuda()
                    embeddings = model.get_day_embeddings(img, layers)

                    # remove batch dim & normalize
                    for layer, embedding in embeddings.items():
                        normalized = unit_normalize(embedding[0]) if unit_norm else embedding[0]
                        embeddings[layer] = normalized.detach().cpu()

                    self.embeddings_day.append(embeddings)

                with open(self.night_files[i], 'rb') as file:
                    img = transform(Image.open(file)).unsqueeze(0)
                    if use_cuda:
                        img = img.cuda()
                    embeddings = model.get_night_embeddings(img, layers)

                    # remove batch dim
                    for layer, embedding in embeddings.items():
                        normalized = unit_normalize(embedding[0]) if unit_norm else embedding[0]
                        embeddings[layer] = normalized.detach().cpu()

                    self.embeddings_night.append(embeddings)

        print('\rDone')

    def __len__(self):
        return min(len(self.embeddings_day), len(self.embeddings_night))

    def __getitem__(self, index):
        return self.embeddings_day[index], self.embeddings_night[index]