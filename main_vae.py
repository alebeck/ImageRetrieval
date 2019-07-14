from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop

from datasets.day_night import DayNightDataset
from models.cycle_vae import CycleVAE
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'paths_day': [
            '../data/synthia_seq4_repacked/day/left/'
        ],
        'paths_night': [
            '../data/synthia_seq4_repacked/night/left/'
        ],
        'transform': Compose([
            CenterCrop(760),
            Resize(16),
            ToTensor()
        ])
    },
    model=CycleVAE,
    model_args={'params': {
        'lr': 1.0e-4,
        'patience': 15,
        'loss_reconst': 10,
        'loss_cycle': 10,
        'loss_kl_reconst': 0.01,
        'loss_kl_cycle': 0.01
    }},
    checkpoint_path=None,
    batch_size=1,
    epochs=10,
    val_size=0.2,
    log_path='log',
    save_every=100
)

trainer = Trainer(config)

print(f'{len(trainer.data.train_loader)} batches')

trainer.train()
