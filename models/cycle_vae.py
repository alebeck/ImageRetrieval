import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.abstract import CustomModule
from models.autoencoder import Autoencoder
from models.decoder import LowerDecoder, UpperDecoder
from models.encoder import UpperEncoder, LowerEncoder


def reconst_loss(x, target):
    return torch.mean(torch.abs(x - target))

def kl_loss(mu):
    mu_2 = torch.pow(mu, 2)
    return torch.mean(mu_2)


class CycleVAE(CustomModule):

    def __init__(self, params: dict):
        self.params = params

        encoder_upper, decoder_lower = UpperEncoder(), LowerDecoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())

        self.optimizer = None
        self.scheduler = None

    def __call__(self, input):
        raise NotImplementedError # TODO

    def init_optimizers(self):
        """
        Is called right before training and after model has been moved to GPU.
        Supposed to initialize optimizers and schedulers.
        """
        params = list(self.ae_day.parameters()) + list(self.ae_night.parameters())
        self.optimizer = Adam([p for p in params if p.requires_grad], lr=self.params['lr'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.params['patience'], verbose=True)

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_sum = 0

        for img_day, img_night in train_loader:
            if use_cuda:
                img_day, img_night = img_day.cuda(), img_night.cuda()

            self.optimizer.zero_grad()

            latent_day, noise_day = self.ae_day.encode(img_day)
            latent_night, noise_night = self.ae_night.encode(img_night)

            # same domain reconstruction
            reconst_day = self.ae_day.decode(latent_day + noise_day)
            reconst_night = self.ae_night.decode(latent_night + noise_night)

            # cross domain
            night_to_day = self.ae_day.decode(latent_night + noise_night)
            day_to_night = self.ae_night.decode(latent_day + noise_day)

            # encode again for cycle loss
            latent_night_to_day, noise_night_to_day = self.ae_day.encode(night_to_day)
            latent_day_to_night, noise_day_to_night = self.ae_night.encode(day_to_night)

            # aaaand decode again
            reconst_cycle_day = self.ae_day.decode(latent_day_to_night + noise_day_to_night)
            reconst_cycle_night = self.ae_night.decode(latent_night_to_day + noise_night_to_day)

            # loss formulations
            loss_reconst_day = reconst_loss(reconst_day, img_day)
            loss_reconst_night = reconst_loss(reconst_night, img_night)
            loss_kl_reconst_day = kl_loss(latent_day)
            loss_kl_reconst_night = kl_loss(latent_night)
            loss_cycle_day = reconst_loss(reconst_cycle_day, img_day)
            loss_cycle_night = reconst_loss(reconst_cycle_night, img_night)
            loss_kl_cycle_day = kl_loss(latent_night_to_day)
            loss_kl_cycle_night = kl_loss(latent_day_to_night)

            # TODO perceptual loss

            loss = \
                self.params['loss_reconst'] * (loss_reconst_day + loss_reconst_night) + \
                self.params['loss_kl_reconst'] * (loss_kl_reconst_day + loss_kl_reconst_night) + \
                self.params['loss_cycle'] * (loss_cycle_day + loss_cycle_night) + \
                self.params['loss_kl_cycle'] * (loss_kl_cycle_day + loss_kl_cycle_night)

            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach().item()

        loss_mean = loss_sum / len(train_loader)
        self.scheduler.step(loss_mean, epoch)

        # log loss
        log_str = f'[Epoch {epoch}] Loss: {loss_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        pass

    def train(self):
        self.ae_day.train()
        self.ae_night.train()

    def eval(self):
        self.ae_day.eval()
        self.ae_night.eval()

    def cuda(self):
        self.ae_day.cuda()
        self.ae_night.cuda()

    def state_dict(self):
        pass

    def optim_state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def load_optim_state_dict(self, state_dict):
        pass
