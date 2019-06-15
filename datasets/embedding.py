import os
from random import choice

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

from models.abstract import CustomModule, EmbeddingGenerator


class EmbeddingDataset(Dataset):

    def __init__(self, model_class, model_args, weights_path, paths_day, paths_night, layers, img_size=128):
        use_cuda = torch.cuda.is_available()

        model: (CustomModule, EmbeddingGenerator) = model_class(**model_args)
        if use_cuda:
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        print("Calculating embeddings... ", end='')

        transform = Compose([
            Resize(img_size),
            ToTensor()
        ])

        self.embeddings_day, self.embeddings_night = [], []

        for path in paths_day:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_day_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = embedding[0]

                    self.embeddings_day.append(embeddings)

        for path in paths_night:
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), 'rb') as file:
                    img = Image.open(file)
                    embeddings = model.get_night_embeddings(transform(img).unsqueeze(0), layers)

                    # remove batch dim
                    for layer, embedding in embeddings.items():
                        embeddings[layer] = embedding[0]

                    self.embeddings_night.append(embeddings)

        print('Done')

    def __len__(self):
        return min(len(self.embeddings_day), len(self.embeddings_night))

    def __getitem__(self, index):
        a = self.embeddings_day[index]
        p = self.embeddings_night[index]
        n = choice(choice([self.embeddings_day, self.embeddings_night]))

        return a, p, n