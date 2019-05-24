import os
import pickle
from datetime import datetime

import torch
from torchvision.transforms import ToPILImage

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

    def train(self):
        log_path = os.path.join(self.config.log_path, str(datetime.now()))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # log config
        with open(os.path.join(log_path, 'config.pickle'), 'wb+') as f:
            pickle.dump(self.config, f)

        # check cuda availability
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('Using GPU...')
            self.model.cuda()

        for epoch in range(self.config.epochs):
            ### TRAINING STEP ###

            # set model to train mode
            self.model.train()

            # train model for one epoch
            info = self.model.train_epoch(self.data.train_loader, epoch, use_cuda)

            # log losses
            log_str = f'[Epoch {epoch}] Train day loss: {info["loss_day"]} Train night loss: {info["loss_night"]}'
            print(log_str)
            with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
                f.write(log_str + '\n')

            ### VALIDATION STEP ###

            # set model to validation mode
            self.model.eval()

            # validate model
            info = self.model.validate(self.data.val_loader, use_cuda)

            # log losses
            log_str = f'[Epoch {epoch}] Val day loss: {info["loss_day"]} Val night loss: {info["loss_night"]}'
            print(log_str)
            with open(os.path.join(log_path, 'log.txt'), 'a+') as f:
                f.write(log_str + '\n')

            # save sample images
            for name, img in info['sample'].items():
                ToPILImage()(img.cpu()).save(os.path.join(log_path, f'{epoch}_{name}.jpeg'), 'JPEG')