import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.custom_module import CustomModule
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=100, verbose=True)  # TODO patience in args

    def train_epoch(self, train_loader, epoch, use_cuda, **kwargs):
        loss_day2night2day_sum, loss_night2day2night_sum = 0, 0

        for day_img, night_img in train_loader:
            if use_cuda:
                day_img, night_img = day_img.cuda(), night_img.cuda()

            # Day -> Night -> Day
            ######################

            self.optimizer.zero_grad()
            # send image through the cycle
            intermediate_latent_1 = self.ae_day.encode(day_img)
            intermediate_night = self.ae_night.decode(intermediate_latent_1)
            intermediate_latent_2 = self.ae_night.encode(intermediate_night)
            out_day = self.ae_day.decode(intermediate_latent_2)
            # optimize
            loss_day2night2day = self.loss_fn(out_day, day_img)
            loss_day2night2day.backward()
            self.optimizer.step()

            # Night -> Day -> Night
            ########################

            self.optimizer.zero_grad()
            # send image through the cycle
            intermediate_latent_1 = self.ae_night.encode(night_img)
            intermediate_day = self.ae_day.decode(intermediate_latent_1)
            intermediate_latent_2 = self.ae_day.encode(intermediate_day)
            out_night = self.ae_night.decode(intermediate_latent_2)
            # optimize
            loss_night2day2night = self.loss_fn(out_night, night_img)
            loss_night2day2night.backward()
            self.optimizer.step()

            ##########################################

            loss_day2night2day_sum += loss_day2night2day
            loss_night2day2night_sum += loss_night2day2night

        loss_day2night2day_mean = loss_day2night2day_sum / len(train_loader)
        loss_night2day2night_mean = loss_night2day2night_sum / len(train_loader)
        loss_mean = (loss_day2night2day_mean + loss_night2day2night_mean) / 2

        self.scheduler.step(loss_mean, epoch)

        # return {'loss day -> night -> day': loss_day2night2day_mean, 'loss night -> day -> night: ': loss_night_mean}
        # return {
        #     'loss_day2night2day_mean': loss_day2night2day_mean,
        #     'loss_night2day2night_mean': loss_night2day2night_mean
        # }
        return {
            'loss_day': loss_day2night2day_mean,
            'loss_night': loss_night2day2night_mean
        }

    def validate(self, val_loader, use_cuda, **kwargs):
        loss_day_sum, loss_night_sum = 0, 0
        day_img, night_img, out_day, out_night = (None,) * 4

        with torch.no_grad():
            for day_img, night_img in val_loader:
                if use_cuda:
                    day_img, night_img = day_img.cuda(), night_img.cuda()

                # TODO implement properly

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
