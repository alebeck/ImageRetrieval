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

        model.eval()

        print("Calculating embeddings... ", end='', flush=True)

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
            for i in idx:
                with open(self.day_files[i], 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_day_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim & normalize
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = unit_normalize(embedding[0]) if unit_norm else embedding[0]

                    self.embeddings_day.append(embeddings)

                with open(self.night_files[i], 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_night_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = unit_normalize(embedding[0]) if unit_norm else embedding[0]

                    self.embeddings_night.append(embeddings)

        print('Done')

    def __len__(self):
        return min(len(self.embeddings_day), len(self.embeddings_night))

    def __getitem__(self, index):
        return self.embeddings_day[index], self.embeddings_night[index]