import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToPILImage

from models.abstract import CustomModule
from models.decoder import LowerDecoder, UpperDecoder
from models.encoder import UpperEncoder, LowerEncoder
from models.autoencoder import Autoencoder


class CycleModel(CustomModule):
    ae_day: Autoencoder
    ae_night: Autoencoder
    reconstruction_loss_factor: float
    cycle_loss_factor: float

    def __init__(self, reconstruction_loss_factor: float, cycle_loss_factor: float):
        # share weights of the upper encoder & lower decoder
        encoder_upper, decoder_lower = UpperEncoder(), LowerDecoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.loss_fn = nn.L1Loss()
        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.cycle_loss_factor = cycle_loss_factor

        self.optimizer = None
        self.scheduler = None

    def __call__(self, input):
        raise NotImplementedError

    def init_optimizers(self):
        """
        Is called right before training and after model has been moved to GPU.
        Supposed to initialize optimizers and schedulers.
        """
        parameters = set()
        parameters |= set(self.ae_day.parameters())
        parameters |= set(self.ae_night.parameters())
        self.optimizer = Adam(parameters)

        # initialize scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=15, verbose=True)

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        loss_day2night2day_sum, loss_night2day2night_sum, loss_day2day_sum, loss_night2night_sum = 0, 0, 0, 0

        for day_img, night_img in train_loader:
            if use_cuda:
                day_img, night_img = day_img.cuda(), night_img.cuda()

            # Day -> Night -> Day
            self.optimizer.zero_grad()
            loss_day2night2day, loss_day2day = self.cycle_plus_reconstruction_loss(day_img, self.ae_day, self.ae_night)
            loss = loss_day2night2day * self.cycle_loss_factor + loss_day2day * self.reconstruction_loss_factor
            loss.backward()
            self.optimizer.step()

            # Night -> Day -> Night
            self.optimizer.zero_grad()
            loss_night2day2night, loss_night2night \
                = self.cycle_plus_reconstruction_loss(night_img, self.ae_night, self.ae_day)
            loss = loss_night2day2night * self.cycle_loss_factor + loss_night2night * self.reconstruction_loss_factor
            loss.backward()
            self.optimizer.step()

            loss_day2night2day_sum += loss_day2night2day
            loss_day2day_sum += loss_day2day
            loss_night2day2night_sum += loss_night2day2night
            loss_night2night_sum += loss_night2night

        loss_day2night2day_mean = loss_day2night2day_sum / len(train_loader)
        loss_day2day_mean = loss_day2day_sum / len(train_loader)
        loss_night2day2night_mean = loss_night2day2night_sum / len(train_loader)
        loss_night2night_mean = loss_night2night_sum / len(train_loader)
        loss_mean = (loss_day2night2day_mean + loss_day2day_mean + loss_night2day2night_mean + loss_night2night_mean)/4

        self.scheduler.step(loss_mean, epoch)

        # log losses
        log_str = f'[Epoch {epoch}] ' \
            f'Train loss day -> night -> day: {loss_day2night2day_mean} ' \
            f'Train loss night -> day -> night: {loss_night2day2night_mean} ' \
            f'Train loss day -> day: {loss_day2day_mean} ' \
            f'Train loss night -> night: {loss_night2night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        loss_day2night2day_sum, loss_night2day2night_sum, loss_day2day_sum, loss_night2night_sum = 0, 0, 0, 0
        day_img, night_img = None, None

        with torch.no_grad():
            for day_img, night_img in val_loader:
                if use_cuda:
                    day_img, night_img = day_img.cuda(), night_img.cuda()

                # Day -> Night -> Day  and  Day -> Day
                loss_day2night2day, loss_day2day = \
                    self.cycle_plus_reconstruction_loss(day_img, self.ae_day, self.ae_night)

                # Night -> Day -> Night  and  Night -> Night
                loss_night2day2night, loss_night2night = \
                    self.cycle_plus_reconstruction_loss(night_img, self.ae_night, self.ae_day)

                loss_day2night2day_sum += loss_day2night2day
                loss_day2day_sum += loss_day2day
                loss_night2day2night_sum += loss_night2day2night
                loss_night2night_sum += loss_night2night

        loss_day2night2day_mean = loss_day2night2day_sum / len(val_loader)
        loss_night2day2night_mean = loss_night2day2night_sum / len(val_loader)
        loss_day2day_mean = loss_day2day_sum / len(val_loader)
        loss_night2night_mean = loss_night2night_sum / len(val_loader)

        # log losses
        log_str = f'[Epoch {epoch}] ' \
            f'Val loss day -> night -> day: {loss_day2night2day_mean} ' \
            f'Val loss night -> day -> night: {loss_night2day2night_mean} ' \
            f'Val loss day -> day: {loss_day2day_mean} ' \
            f'Val loss night -> night: {loss_night2night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

        # create sample images

        latent_day = self.ae_day.encode(day_img[0].unsqueeze(0))
        latent_night = self.ae_night.encode(night_img[0].unsqueeze(0))
        # reconstruction
        day2day = self.ae_day.decode(latent_day)
        night2night = self.ae_night.decode(latent_night)
        # domain translation
        day2night = self.ae_night.decode(latent_day)
        night2day = self.ae_day.decode(latent_night)
        # cycle
        day2night2day = self.ae_day.decode(self.ae_night.encode(day2night))
        night2day2night = self.ae_night.decode(self.ae_day.encode(night2day))

        # save sample images
        samples = {
            'day_img': day_img[0],
            'night_img': night_img[0],
            'day2day': day2day[0],
            'night2night': night2night[0],
            'day2night': day2night[0],
            'night2day': night2day[0],
            'day2night2day': day2night2day[0],
            'night2day2night': night2day2night[0],
        }

        for name, img in samples.items():
            ToPILImage()(img.cpu()).save(os.path.join(log_path, f'{epoch}_{name}.jpeg'), 'JPEG')

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
        return cycle_loss, reconstruction_loss

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

    def optim_state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.ae_day.encoder_lower.load_state_dict(state['encoder_lower_day'])
        self.ae_night.encoder_lower.load_state_dict(state['encoder_lower_night'])
        self.ae_day.encoder_upper.load_state_dict(state['encoder_upper'])
        self.ae_day.decoder.load_state_dict(state['decoder_day'])
        self.ae_night.decoder.load_state_dict(state['decoder_night'])
