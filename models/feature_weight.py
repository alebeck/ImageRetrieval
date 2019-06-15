import torch
from torch.optim import Adam

from models.abstract import CustomModule


class FeatureWeight(CustomModule):

    def __call__(self, input):
        raise NotImplementedError #TODO

    def __init__(self, layers: dict, margin:float = 1.0):
        """
        :param layers: Layers to use and their number of channels, e.g. {'conv1': 512, 'conv2': 256}
        """
        self.layers = layers
        self.margin = torch.tensor(margin).float()

        self.weights = {
            layer: torch.ones(size=(1, size)).float().t().requires_grad_() for layer, size in layers.items()
        }

        self.optimizer = Adam(self.weights.values())

    def calculate_loss(self, a, p, n):
        # weight channels of a, p and n
        ap_dist_sum, an_dist_sum = torch.tensor(0).float(), torch.tensor(0).float()
        for layer, size in self.layers.items():
            assert a[layer].shape[1] == p[layer].shape[1] == n[layer].shape[1] == self.layers[layer]

            B, C = a[layer].shape[:2]

            # normalize activations in channel dim
            # TODO

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
        return torch.max(ap_dist - an_dist + self.margin, torch.tensor(0.).float())

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for a, p, n in train_loader:
            if use_cuda:
                a, p, n = a.cuda(), p.cuda(), n.cuda()

            self.optimizer.zero_grad()

            loss = self.calculate_loss(a, p, n)
            loss.requires_grad_().backward()

            self.optimizer.step()

            loss_sum += loss

        loss_mean = loss_sum / len(train_loader)

        print(f'Train loss: {loss_mean}')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for a, p, n in val_loader:
            if use_cuda:
                a, p, n = a.cuda(), p.cuda(), n.cuda()

            loss = self.calculate_loss(a, p, n)
            loss_sum += loss

        loss_mean = loss_sum / len(val_loader)

        print(f'Val loss: {loss_mean}')

    def train(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        self.weights = self.weights.cuda()

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass