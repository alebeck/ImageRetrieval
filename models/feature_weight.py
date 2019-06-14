import torch
from torch.optim import Adam

from models.abstract import CustomModule


class FeatureWeight(CustomModule):

    def __init__(self, layers: dict, margin:float = 1.0):
        """
        :param layers: Layers to use and their number of channels, e.g. {'conv1': 512, 'conv2': 256}
        """
        self.layers = layers
        self.margin = torch.tensor(margin).float()

        self.weights = {
            layer: torch.ones(size=(1, size)).float().t().requires_grad_() for layer, size in layers.items()
        }

        self.optimizer = Adam([self.weights['conv1']])

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for a, p, n in train_loader:
            if use_cuda:
                a, p, n = a.cuda(), p.cuda(), n.cuda()

            self.optimizer.zero_grad()

            # weight channels of a, p and n # TODO normalize
            ap_dists, an_dists = [], []
            for layer, size in self.layers.items():
                assert a[layer].shape[1] == p[layer].shape[1] == n[layer].shape[1] == self.layers[layer]

                a_weighted = a[layer] * self.weights[layer]
                p_weighted = p[layer] * self.weights[layer]
                n_weighted = n[layer] * self.weights[layer]

                # average L2 norm across spatial dimension
                ap_dists.append(torch.norm(a_weighted - p_weighted, 2, dim=1).mean())
                an_dists.append(torch.norm(a_weighted - n_weighted, 2, dim=1).mean())

            # average across layer dimension
            ap_dist = torch.mean(torch.tensor(ap_dists).float())
            an_dist = torch.mean(torch.tensor(an_dists).float())

            # triplet loss
            loss = torch.max(ap_dist - an_dist + self.margin, torch.tensor(0.).float())

            loss.requires_grad_().backward()
            self.optimizer.step()

            loss_sum += loss

        loss_mean = loss_sum / len(train_loader)

        print(f'Train loss: {loss_mean}')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        #for a, p, n in val_loader:
        #    if use_cuda:
        #        a, p, n = a.cuda(), p.cuda(), n.cuda()

        #    loss = self.loss_fn(a, p, n)
        #    loss_sum += loss

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