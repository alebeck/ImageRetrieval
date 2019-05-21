import os
import pickle

from utils.config import TrainingConfig
from utils.data import DataSplitter


class Trainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data = DataSplitter(
            dataset=config.dataset(**config.dataset_args),
            batch_size=config.batch_size,
            val_size=config.val_size
        )
        self.model = config.model(**config.model_args)

        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

        # log: indicate start of new training process
        with open(os.path.join(self.config.log_path, 'log.txt'), 'a+') as f:
            f.write(f'\n\n--------------------------------------------------------------------------------------\n\n')

        # log config
        # TODO this does nothing apparently
        with open(os.path.join(config.log_path, 'config.pickle'), 'ab+') as f:
            pickle.dump(config, f)

    def train(self):
        for epoch in range(self.config.epochs):
            # set model to train mode
            self.model.train()

            # get training data loader
            train_loader = self.data.train_loader

            # train model for one epoch
            info = self.model.train_epoch(train_loader)

            # log results
            with open(os.path.join(self.config.log_path, 'log.txt'), 'a+') as f:
                f.write(f'[Epoch {epoch}] Train day loss: {info["loss_day"]} Train night loss: {info["loss_night"]}\n')

            # set model to validation mode
            self.model.eval()

            # get validation data loader
            val_loader = self.data.val_loader

            # validate model
            info = self.model.validate(val_loader)

            # log results
            with open(os.path.join(self.config.log_path, 'log.txt'), 'a+') as f:
                f.write(f'[Epoch {epoch}] Val day loss: {info["loss_day"]} Val night loss: {info["loss_night"]}\n')
