import os

import torch
from torch.optim import Adam

from models.abstract import CustomModule
from utils.functions import select_triplets


class FeatureWeight(CustomModule):

    def __init__(self, layers: dict, margin: float = 1.0):
        """
        :param layers: Layers to use and their number of channels, e.g. {'conv1': 512, 'conv2': 256}
        """
        self.layers = layers
        self.margin = torch.tensor(margin).float()

        self.weights = {
            layer: torch.ones(size=(1, size)).float().t().requires_grad_() for layer, size in layers.items()
        }

        self.optimizer = Adam(self.weights.values())

    def __call__(self, input):
        raise NotImplementedError  # TODO

    def calculate_loss(self, a, p, n, use_cuda):
        # weight channels of a, p and n
        ap_dist_sum, an_dist_sum = torch.tensor(0).float(), torch.tensor(0).float()

        if use_cuda:
            ap_dist_sum, an_dist_sum = ap_dist_sum.cuda(), an_dist_sum.cuda()

        for layer, size in self.layers.items():
            assert a[layer].shape[1] == p[layer].shape[1] == n[layer].shape[1] == self.layers[layer]

            B, C = a[layer].shape[:2]

            # weight activations
            a_weighted = a[layer].view(B, C, -1) * self.weights[layer]
            p_weighted = p[layer].view(B, C, -1) * self.weights[layer]
            n_weighted = n[layer].view(B, C, -1) * self.weights[layer]

            # average L2 norm across spatial dimension
            ap_dist_sum += torch.norm(a_weighted - p_weighted, 2, dim=1).mean()
            an_dist_sum += torch.norm(a_weighted - n_weighted, 2, dim=1).mean()

        # average across layer dimension
        ap_dist = ap_dist_sum / len(self.layers)
        an_dist = an_dist_sum / len(self.layers)

        # triplet loss
        zero = torch.tensor(0.).float()
        if use_cuda:
            zero = zero.cuda()

        return torch.max(ap_dist - an_dist + self.margin, zero)

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for embeddings_day, embeddings_night in train_loader:
            if use_cuda:
                for layer in self.layers:
                    embeddings_day[layer], embeddings_night[layer] = embeddings_day[layer].cuda(), embeddings_night[
                        layer].cuda()

            self.optimizer.zero_grad()

            loss = self.calculate_loss(*select_triplets(embeddings_day, embeddings_night), use_cuda)
            loss.requires_grad_().backward()

            self.optimizer.step()

            loss_sum += loss

        loss_mean = loss_sum / len(train_loader)

        log_str = f'[Epoch {epoch}] Train loss: {loss_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for embeddings_day, embeddings_night in val_loader:
            if use_cuda:
                for layer in self.layers:
                    embeddings_day[layer], embeddings_night[layer] = embeddings_day[layer].cuda(), embeddings_night[
                        layer].cuda()

            loss = self.calculate_loss(*select_triplets(embeddings_day, embeddings_night), use_cuda)
            loss_sum += loss

        loss_mean = loss_sum / len(val_loader)

        log_str = f'[Epoch {epoch}] Val loss: {loss_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def train(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        for layer in self.layers:
            self.weights[layer] = self.weights[layer].cuda()

    def state_dict(self):
        return self.weights

    def load_state_dict(self, state_dict):
        self.weights = state_dict