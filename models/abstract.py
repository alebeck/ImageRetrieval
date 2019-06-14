from abc import abstractmethod

import torch


class CustomModule:
    """
    Superclass for models which are intended to be used with our training framework
    """

    @abstractmethod
    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def cuda(self):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError


class EmbeddingGenerator:
    """
    Every model which is supposed to be used with triplet learning should be a subclass of this
    """

    @abstractmethod
    def get_day_embeddings(self, img: torch.FloatTensor):
        raise NotImplementedError

    @abstractmethod
    def get_night_embeddings(self, img: torch.FloatTensor):
        raise NotImplementedError