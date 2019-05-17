from utils.config import TrainingConfig
from utils.data import DataSplitter


class Trainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data = DataSplitter(
            dataset=self.config.dataset,
            batch_size=config.batch_size,
            val_size=config.val_size
        )
        self.model = config.model(**config.model_args)

        # TODO Setup logging

    def train(self):
        for epoch in range(self.config.epochs):
            # set model to train mode
            self.model.train()

            # get training data loader
            train_loader = self.data.train_loader

            # train model for one epoch
            info = self.model.train_epoch(train_loader, self.config.batch_size, self.config.optim, self.config.optim_args)

            # TODO Log results

            # set model to validation mode
            self.model.eval()

            # get validation data loader
            val_loader = self.data.val_loader

            # validate model
            info = self.model.validate(val_loader, self.config.batch_size)

            # TODO Log results
