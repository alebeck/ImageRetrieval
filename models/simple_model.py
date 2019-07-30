import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToPILImage

from models.abstract import CustomModule, EmbeddingGenerator
from models.decoder import LowerDecoder, UpperDecoder
from models.encoder import LowerEncoder, UpperEncoder
from models.autoencoder import Autoencoder


class SimpleModel(CustomModule, EmbeddingGenerator):

    ae_day: Autoencoder
    ae_night: Autoencoder

    def __init__(self):
        encoder_upper, decoder_lower = UpperEncoder(), LowerDecoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.loss_fn = nn.L1Loss()

        self.optimizer_day = None
        self.optimizer_night = None
        self.scheduler_day = None
        self.scheduler_night = None

    def __call__(self, input):
        raise NotImplementedError

    def init_optimizers(self):
        """
        Is called right before training and after model has been moved to GPU.
        Supposed to initialize optimizers and schedulers.
        """
        self.optimizer_day = Adam(self.ae_day.parameters(), lr=1e-4)
        self.optimizer_night = Adam(self.ae_night.parameters(), lr=1e-4)
        self.scheduler_day = ReduceLROnPlateau(self.optimizer_day, patience=15, verbose=True)
        self.scheduler_night = ReduceLROnPlateau(self.optimizer_night, patience=15, verbose=True)

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
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

        # log losses
        log_str = f'[Epoch {epoch}] Train day loss: {loss_day_mean} Train night loss: {loss_night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
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

        # log losses
        log_str = f'[Epoch {epoch}] Val day loss: {loss_day_mean} Val night loss: {loss_night_mean}'
        print(log_str)
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
            f.write(log_str + '\n')

        # save sample images
        samples = {
            'day_img': day_img[0],
            'night_img': night_img[0],
            'out_day': out_day[0],
            'out_night': out_night[0],
            'day_to_night': day_to_night[0],
            'night_to_day': night_to_day[0]
        }

        for name, img in samples.items():
            ToPILImage()(img.cpu()).save(os.path.join(log_path, f'{epoch}_{name}.jpeg'), 'JPEG')

    def register_hooks(self, layers): # TODO put this and the next method in context manager
        """
        This function is not supposed to be called from outside the class.
        """
        handles = []
        embedding_dict = {}

        def get_hook(name, embedding_dict):
            def hook(model, input, output):
                embedding_dict[name] = output.detach()
            return hook

        for layer in layers:
            hook = get_hook(layer, embedding_dict)
            handles.append(getattr(self.ae_day.encoder_upper, layer).register_forward_hook(hook))

        return handles, embedding_dict

    def deregister_hooks(self, handles):
        """
        This function is not supposed to be called from outside the class.
        """
        for handle in handles:
            handle.remove()

    def get_day_embeddings(self, img, layers):
        """
        Returns deep embeddings for the passed layers inside the upper encoder.
        """
        handles, embedding_dict = self.register_hooks(layers)

        # forward pass
        self.ae_day.encode(img)

        self.deregister_hooks(handles)

        return embedding_dict

    def get_night_embeddings(self, img, layers):
        """
        Returns deep embeddings for the passed layers inside the upper encoder.
        """
        handles, embedding_dict = self.register_hooks(layers)

        # forward pass
        self.ae_night.encode(img)

        self.deregister_hooks(handles)

        return embedding_dict

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
            'optimizer_day': self.optimizer_day.state_dict(),
            'optimizer_night': self.optimizer_night.state_dict()
        }

    def load_state_dict(self, state):
        self.ae_day.encoder_lower.load_state_dict(state['encoder_lower_day'])
        self.ae_night.encoder_lower.load_state_dict(state['encoder_lower_night'])
        self.ae_day.encoder_upper.load_state_dict(state['encoder_upper'])
        self.ae_day.decoder.load_state_dict(state['decoder_day'])
        self.ae_night.decoder.load_state_dict(state['decoder_night'])

    def load_optim_state_dict(self, state):
        self.optimizer_day.load_state_dict(state['optimizer_day'])
        self.optimizer_night.load_state_dict(state['optimizer_night'])
