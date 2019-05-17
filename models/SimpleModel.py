from models.custom_module import CustomModule
from models.decoder import Decoder
from models.encoder import Encoder
from models.encoder_decoder_pair import EncoderDecoderPair

import torch.nn as nn
from torch.optim import Adam


class SimpleModel(CustomModule):
    enc_dec_day: EncoderDecoderPair
    enc_dec_night: EncoderDecoderPair

    def __init__(self):
        self.encoder = Encoder() # TODO pretrained
        self.enc_dec_day = EncoderDecoderPair(self.encoder, Decoder())
        self.enc_dec_night = EncoderDecoderPair(self.encoder, Decoder())
        self.loss_fn = nn.L1Loss()  # TODO Which loss?

        self.optimizer_day = Adam(self.enc_dec_day.parameters())  # TODO put args in config (lr, weight_decay)
        self.optimizer_night = Adam(self.enc_dec_night.parameters())  # TODO put args in config (lr, weight_decay)

    def train_epoch(self, train_loader, batch_size, **kwargs):

        loss_day_sum, loss_night_sum = 0, 0

        for (day_img, night_img) in train_loader:

            # zero day gradients
            self.optimizer_day.zero_grad()

            # train first pair
            out_day = self.enc_dec_day(day_img)
            loss_day = self.loss_fn(out_day, day_img)

            # optimize
            loss_day.backward()
            self.optimizer_day.step()

            # zero night gradients
            self.optimizer_night.zero_grad()

            # train first pair
            out_night = self.enc_dec_night(night_img)
            loss_night = self.loss_fn(out_night, night_img)

            # optimize
            loss_night.backward()
            self.optimizer_night.step()

            loss_day_sum += loss_day
            loss_night_sum += loss_night

        loss_day_mean = loss_day_sum / len(train_loader)
        loss_night_mean = loss_night_sum / len(train_loader)

        return {'loss_day': loss_day_mean, 'loss_night': loss_night_mean}

    def validate(self, val_loader, batch_size, **kwargs):
        pass # TODO validation pass
