import os
import pickle
from datetime import datetime

import torch

from utils.config import TrainingConfig
from utils.data import DataSplitter


class Trainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data = DataSplitter(
            dataset=config.dataset(**config.dataset_args),
            batch_size=config.batch_size,
            val_size=config.val_size,
            shuffle=False
        )
        self.model = config.model(**config.model_args)
        self.epoch_start = 0
        self.checkpoint = None

        # load checkpoint
        if self.config.checkpoint_path is not None:
            print('Resuming training from checkpoint...')
            self.checkpoint = torch.load(self.config.checkpoint_path)
            self.model.load_state_dict(self.checkpoint['model'])
            self.epoch_start = self.checkpoint['epoch']

        # check cuda availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('Using GPU...')
            self.model.cuda()

        self.model.init_optimizers()

        # resume optimizer state
        if self.checkpoint is not None:
            self.model.load_optim_state_dict(self.checkpoint['optimizer'])

    def train(self):
        log_path = os.path.join(self.config.log_path, str(datetime.now()))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # log config
        with open(os.path.join(log_path, 'config.pickle'), 'wb+') as f:
            pickle.dump(self.config, f)

        for epoch in range(self.epoch_start, self.config.epochs):
            ### TRAINING STEP ###

            # set model to train mode
            self.model.train()

            # train model for one epoch
            self.model.train_epoch(self.data.train_loader, epoch, self.use_cuda, log_path)

            ### VALIDATION STEP ###

            # set model to validation mode
            self.model.eval()

            # validate model
            self.model.validate(self.data.val_loader, epoch, self.use_cuda, log_path)

            # save checkpoint
            if epoch % self.config.save_every == 0:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.model.optim_state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(checkpoint, os.path.join(log_path, f'{epoch}_weights.pt'))
