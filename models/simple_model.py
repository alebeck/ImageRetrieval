import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.custom_module import CustomModule
from models.decoder import Decoder
from models.encoder import LowerEncoder, UpperEncoder
from models.autoencoder import Autoencoder


class SimpleModel(CustomModule):
    ae_day: Autoencoder
    ae_night: Autoencoder

    def __init__(self):
        # share weights of the upper encoder stage
        encoder_upper = UpperEncoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, Decoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, Decoder())
        self.loss_fn = nn.L1Loss()  # TODO Which loss?

        self.optimizer_day = Adam(self.ae_day.parameters())  # TODO put args in config (lr, weight_decay)
        self.optimizer_night = Adam(self.ae_night.parameters())  # TODO put args in config (lr, weight_decay)

        # initialize scheduler
        self.scheduler_day = ReduceLROnPlateau(self.optimizer_day, patience=100, verbose=True)  # TODO patience in args
        self.scheduler_night = ReduceLROnPlateau(self.optimizer_night, patience=100, verbose=True)  # TODO patience in args

    def train_epoch(self, train_loader, epoch, use_cuda, **kwargs):
        loss_day_sum, loss_night_sum = 0, 0

        for day_img, night_img in train_loader:
            if use_cuda:
                day_img, night_img = day_img.cuda(), night_img.cuda()

            # zero day gradients
            self.optimizer_day.zero_grad()

            # train day autoencoder
            out_day = self.ae_day(day_img)
            loss_day = self.loss_fn(out_day, day_img)

            # optimize
            loss_day.backward()
            self.optimizer_day.step()

            # zero night gradients
            self.optimizer_night.zero_grad()

            # train night autoencoder
            out_night = self.ae_night(night_img)
            loss_night = self.loss_fn(out_night, night_img)

            # optimize
            loss_night.backward()
            self.optimizer_night.step()

            loss_day_sum += loss_day
            loss_night_sum += loss_night

        loss_day_mean = loss_day_sum / len(train_loader)
        loss_night_mean = loss_night_sum / len(train_loader)

        self.scheduler_day.step(loss_day_mean, epoch)
        self.scheduler_night.step(loss_night_mean, epoch)

        return {'loss_day': loss_day_mean, 'loss_night': loss_night_mean}

    def validate(self, val_loader, use_cuda, **kwargs):
        loss_day_sum, loss_night_sum = 0, 0
        day_img, night_img, out_day, out_night = (None,) * 4

        with torch.no_grad():
            for day_img, night_img in val_loader:
                if use_cuda:
                    day_img, night_img = day_img.cuda(), night_img.cuda()

                out_day = self.ae_day(day_img)
                loss_day = self.loss_fn(out_day, day_img)

                out_night = self.ae_night(night_img)
                loss_night = self.loss_fn(out_night, night_img)

                loss_day_sum += loss_day
                loss_night_sum += loss_night

        loss_day_mean = loss_day_sum / len(val_loader)
        loss_night_mean = loss_night_sum / len(val_loader)

        # domain translation
        day_to_night = self.ae_night.decode(self.ae_day.encode(day_img[0].unsqueeze(0)))
        night_to_day = self.ae_day.decode(self.ae_night.encode(night_img[0].unsqueeze(0)))

        return {
            'loss_day': loss_day_mean,
            'loss_night': loss_night_mean,
            'sample': {
                'day_img': day_img[0],
                'night_img': night_img[0],
                'out_day': out_day[0],
                'out_night': out_night[0],
                'day_to_night': day_to_night[0],
                'night_to_day': night_to_day[0]
            }
        }

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
