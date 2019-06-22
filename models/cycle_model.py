import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.abstract import CustomModule
from models.decoder import Decoder
from models.encoder import LowerEncoder, UpperEncoder
from models.autoencoder import Autoencoder


class CycleModel(CustomModule):
    ae_day: Autoencoder
    ae_night: Autoencoder

    def __init__(self):
        # share weights of the upper encoder stage
        encoder_upper = UpperEncoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, Decoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, Decoder())
        self.loss_fn = nn.L1Loss()  # TODO Which loss?

        # TODO is there a nicer way to write this?
        parameters = set()
        parameters |= set(self.ae_day.parameters())
        parameters |= set(self.ae_night.parameters())
        self.optimizer = Adam(parameters)  # TODO put args in config (lr, weight_decay)

        # initialize scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=15, verbose=True)  # TODO patience in args

    def __call__(self, input):
        raise NotImplementedError  # TODO

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_day2night2day_sum, loss_night2day2night_sum = 0, 0

        for day_img, night_img in train_loader:
            if use_cuda:
                day_img, night_img = day_img.cuda(), night_img.cuda()

            # Day -> Night -> Day
            self.optimizer.zero_grad()
            loss_day2night2day = self.cycle_plus_reconstruction_loss(day_img, self.ae_day, self.ae_night)
            loss_day2night2day.backward()
            self.optimizer.step()

            # Night -> Day -> Night
            self.optimizer.zero_grad()
            loss_night2day2night = self.cycle_plus_reconstruction_loss(night_img, self.ae_night, self.ae_day)
            loss_night2day2night.backward()
            self.optimizer.step()

            loss_day2night2day_sum += loss_day2night2day
            loss_night2day2night_sum += loss_night2day2night

        loss_day2night2day_mean = loss_day2night2day_sum / len(train_loader)
        loss_night2day2night_mean = loss_night2day2night_sum / len(train_loader)
        loss_mean = (loss_day2night2day_mean + loss_night2day2night_mean) / 2

        self.scheduler.step(loss_mean, epoch)

        # log losses
        log_str = f'[Epoch {epoch}] ' \
            f'Train loss day -> night -> day: {loss_day2night2day_mean} ' \
            f'Train loss night -> day -> night: {loss_night2day2night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        loss_day2night2day_sum, loss_night2day2night_sum = 0, 0
        # TODO: use this for logging
        # day_img, night_img, out_day, out_night = (None,) * 4

        with torch.no_grad():
            for day_img, night_img in val_loader:
                if use_cuda:
                    day_img, night_img = day_img.cuda(), night_img.cuda()

                # Day -> Night -> Day
                loss_day2night2day_sum += self.cycle_plus_reconstruction_loss(day_img, self.ae_day, self.ae_night)
                # Night -> Day -> Night
                loss_night2day2night_sum += self.cycle_plus_reconstruction_loss(night_img, self.ae_night, self.ae_day)

        loss_day2night2day_mean = loss_day2night2day_sum / len(val_loader)
        loss_night2day2night_mean = loss_night2day2night_sum / len(val_loader)

        # domain translation
        # TODO: use this for logging
        # day_to_night = self.ae_night.decode(self.ae_day.encode(day_img[0].unsqueeze(0)))
        # night_to_day = self.ae_day.decode(self.ae_night.encode(night_img[0].unsqueeze(0)))

        # log losses
        log_str = f'[Epoch {epoch}] ' \
            f'Val loss day -> night -> day: {loss_day2night2day_mean} ' \
            f'Val loss night -> day -> night: {loss_night2day2night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

        # TODO: save sample images

    def cycle_plus_reconstruction_loss(self, image, autoencoder1, autoencoder2):
        # send the image through the cycle
        intermediate_latent_1 = autoencoder1.encode(image)
        intermediate_opposite = autoencoder2.decode(intermediate_latent_1)
        intermediate_latent_2 = autoencoder2.encode(intermediate_opposite)
        cycle_img = autoencoder1.decode(intermediate_latent_2)

        # do simple reconstruction
        reconstructed_img = autoencoder1.decode(intermediate_latent_1)

        cycle_loss = self.loss_fn(cycle_img, image)
        reconstruction_loss = self.loss_fn(reconstructed_img, image)
        return cycle_loss + reconstruction_loss

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
        return {
            'encoder_lower_day': self.ae_day.encoder_lower.state_dict(),
            'encoder_lower_night': self.ae_night.encoder_lower.state_dict(),
            'encoder_upper': self.ae_day.encoder_upper.state_dict(),
            'decoder_day': self.ae_day.decoder.state_dict(),
            'decoder_night': self.ae_night.decoder.state_dict()
        }

    def load_state_dict(self, state):
        self.ae_day.encoder_lower.load_state_dict(state['encoder_lower_day'])
        self.ae_night.encoder_lower.load_state_dict(state['encoder_lower_night'])
        self.ae_day.encoder_upper.load_state_dict(state['encoder_upper'])
        self.ae_day.decoder.load_state_dict(state['decoder_day'])
        self.ae_night.decoder.load_state_dict(state['decoder_night'])
