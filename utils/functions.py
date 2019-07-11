import random

import torch


def select_triplets(embeddings_day, embeddings_night):
    """
    Builds triplets from given day and night embeddings
    :param embeddings_day:
    :param embeddings_night:
    :return:
    """
    negatives = {layer: [] for layer in embeddings_day.keys()}
    batch_size = len(list(embeddings_day.values())[0])

    for i in range(batch_size):
        n_dataset = random.choice([embeddings_day, embeddings_night])
        n_index = random.choice(list(range(i)) + list(range(i + 1, batch_size)))
        for layer in embeddings_day.keys():
            negatives[layer].append(n_dataset[layer][n_index])

    negatives = {layer: torch.stack(embs) for layer, embs in negatives.items()}

    return embeddings_day, embeddings_night, negatives

def unit_normalize(x, dim=0):
    """
    Unit-normalizes a tensor along a given dimension
    :param x:
    :param dim:
    :return:
    """
    return (x - x.mean(dim=dim)) / (x.std(dim=dim) + torch.tensor(1e-8).float())