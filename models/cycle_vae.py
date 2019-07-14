from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.abstract import CustomModule
from models.autoencoder import Autoencoder
from models.decoder import LowerDecoder, UpperDecoder
from models.encoder import UpperEncoder, LowerEncoder


class CycleVAE(CustomModule):

    def __init__(self):
        encoder_upper, decoder_lower = UpperEncoder(), LowerDecoder()
        self.ae_day = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())
        self.ae_night = Autoencoder(LowerEncoder(), encoder_upper, decoder_lower, UpperDecoder())

        self.optimizer_day = None
        self.optimizer_night = None
        self.scheduler_day = None
        self.scheduler_night = None

    def __call__(self, input):
        raise NotImplementedError # TODO

    def init_optimizers(self):
        """
        Is called right before training and after model has been moved to GPU.
        Supposed to initialize optimizers and schedulers.
        """
        self.optimizer_day = Adam(self.ae_day.parameters(), lr=1e-4)  # TODO put args in config (lr, weight_decay)
        self.optimizer_night = Adam(self.ae_night.parameters(), lr=1e-4)  # TODO put args in config (lr, weight_decay)
        self.scheduler_day = ReduceLROnPlateau(self.optimizer_day, patience=15, verbose=True)  # TODO patience in args
        self.scheduler_night = ReduceLROnPlateau(self.optimizer_night, patience=15, verbose=True)  # TODO patience in args

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
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