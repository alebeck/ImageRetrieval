from abc import abstractmethod


class CustomModule:
    """
    Superclass for models which are intended to be used with our training framework
    """

    @abstractmethod
    def train_epoch(self, train_loader, epoch, use_cuda, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, val_loader, use_cuda, **kwargs):
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
