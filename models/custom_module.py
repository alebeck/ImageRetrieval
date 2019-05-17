from abc import abstractmethod


class CustomModule:
    """
    Superclass for models which are intended to be used with our training framework
    """

    @abstractmethod
    def train_epoch(self, train_loader, batch_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, val_loader, batch_size, **kwargs):
        raise NotImplementedError
