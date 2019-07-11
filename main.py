from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop

from datasets.day_night import DayNightDataset
from models.cycle_model import CycleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'paths_day': [
            'data/synthia_seq4_repacked/day/left',
            'data/synthia_seq4_repacked/day/right',
        ],
        'paths_night': [
            'data/synthia_seq4_repacked/night/left',
            'data/synthia_seq4_repacked/night/right',
        ],
        'transform': Compose([
            CenterCrop(760),
            Resize(128),
            ToTensor()
        ])
    },
    model=CycleModel,
    checkpoint_path=None,
    model_args={
        'reconstruction_loss_factor': 1.,
        'cycle_loss_factor': 1.,
    },
    batch_size=64,
    epochs=30,
    val_size=0.2,
    log_path='log',
    save_every=100
)

trainer = Trainer(config)

print(f'{len(trainer.data.train_loader)} batches')

# step 1: train with reconstruction loss
trainer.train()

# step 2: train with cycle loss
trainer.model.reconstruction_loss_factor = 0.
trainer.model.cycle_loss_factor = 1.
trainer.train()
