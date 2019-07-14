import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from models.abstract import CustomModule, EmbeddingGenerator
from utils.functions import unit_normalize


class EmbeddingDataset(Dataset):

    def __init__(self, model_class, model_args, weights_path, paths_day, paths_night, layers, transform=None):
        use_cuda = torch.cuda.is_available()

        model: (CustomModule, EmbeddingGenerator) = model_class(**model_args)
        if weights_path is not None:
            if use_cuda:
                model.load_state_dict(torch.load(weights_path)['model'])
            else:
                model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])

        print("Calculating embeddings... ", end='', flush=True)

        if transform is None:
            transform = ToTensor()

        self.embeddings_day, self.embeddings_night = [], []

        for path in paths_day:
            for filename in sorted(os.listdir(path)):
                if filename.startswith('.'):
                    continue

                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_day_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim & normalize
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = unit_normalize(embedding[0])

                    self.embeddings_day.append(embeddings)

        for path in paths_night:
            for filename in sorted(os.listdir(path)):
                if filename.startswith('.'):
                    continue

                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_night_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = unit_normalize(embedding[0])

                    self.embeddings_night.append(embeddings)

        print('Done')

    def __len__(self):
        return min(len(self.embeddings_day), len(self.embeddings_night))

    def __getitem__(self, index):
        return self.embeddings_day[index], self.embeddings_night[index]