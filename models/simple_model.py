from models.custom_module import CustomModule
from models.decoder import Decoder
from models.encoder import Encoder
from models.autoencoder import Autoencoder

import torch
import torch.nn as nn
from torch.optim import Adam


class SimpleModel(CustomModule):
    autoencoder_day: Autoencoder
    autoencoder_night: Autoencoder

    def __init__(self):
        self.encoder = Encoder()  # TODO pre-trained (later in project)
        self.autoencoder_day = Autoencoder(self.encoder, Decoder())
        self.autoencoder_night = Autoencoder(self.encoder, Decoder())
        self.loss_fn = nn.L1Loss()  # TODO Which loss?

        self.optimizer_day = Adam(self.autoencoder_day.parameters())  # TODO put args in config (lr, weight_decay)
        self.optimizer_night = Adam(self.autoencoder_night.parameters())  # TODO put args in config (lr, weight_decay)

    def train_epoch(self, train_loader, **kwargs):
        loss_day_sum, loss_night_sum = 0, 0

        for (day_img, night_img) in train_loader:
            # zero day gradients
            self.optimizer_day.zero_grad()

            # train day autoencoder
            out_day = self.autoencoder_day(day_img)
            loss_day = self.loss_fn(out_day, day_img)

            # optimize
            loss_day.backward()
            self.optimizer_day.step()

            # zero night gradients
            self.optimizer_night.zero_grad()

            # train night autoencoder
            out_night = self.autoencoder_night(night_img)
            loss_night = self.loss_fn(out_night, night_img)

            # optimize
            loss_night.backward()
            self.optimizer_night.step()

            loss_day_sum += loss_day
            loss_night_sum += loss_night

        loss_day_mean = loss_day_sum / len(train_loader)
        loss_night_mean = loss_night_sum / len(train_loader)

        return {'loss_day': loss_day_mean, 'loss_night': loss_night_mean}

    def validate(self, val_loader, **kwargs):
        loss_day_sum, loss_night_sum = 0, 0
        day_img, night_img, out_day, out_night = (None,) * 4

        with torch.no_grad():
            for (day_img, night_img) in val_loader:
                out_day = self.autoencoder_day(day_img)
                loss_day = self.loss_fn(out_day, day_img)

                out_night = self.autoencoder_night(night_img)
                loss_night = self.loss_fn(out_night, night_img)

                loss_day_sum += loss_day
                loss_night_sum += loss_night

        loss_day_mean = loss_day_sum / len(val_loader)
        loss_night_mean = loss_night_sum / len(val_loader)

        return {
            'loss_day': loss_day_mean,
            'loss_night': loss_night_mean,
            'sample': {
                'day_img': day_img[0],
                'night_img': night_img[0],
                'out_day': out_day[0],
                'out_night': out_night[0]
            }
        }

    def train(self):
        self.encoder.train()
        self.autoencoder_day.decoder.train()
        self.autoencoder_night.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.autoencoder_day.decoder.eval()
        self.autoencoder_night.decoder.eval()
