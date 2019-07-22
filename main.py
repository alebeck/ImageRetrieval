from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop

from datasets.day_night import DayNightDataset
from models.simple_model import SimpleModel
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
    model=SimpleModel,
    checkpoint_path=None,
    model_args={},
    batch_size=32,
    epochs=71,
    val_size=0.2,
    log_path='drive/My Drive/adl4cv/training-logs/baseline-reconstruction-loss',
    save_every=5
)

trainer = Trainer(config)

print(f'{len(trainer.data.train_loader)} batches')

trainer.train()
