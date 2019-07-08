from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop

from datasets.day_night import DayNightDataset
from models.simple_model import SimpleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'paths_day': [],
        'paths_night': [],
        'transform': Compose([
            CenterCrop(760),
            Resize(128),
            ToTensor()
        ])
    },
    model=SimpleModel,
    checkpoint_path=None,
    model_args={},
    batch_size=1,
    epochs=10,
    val_size=0.2,
    log_path='log',
    save_every=100
)

trainer = Trainer(config)

print(f'{len(trainer.data.train_loader)} batches')

trainer.train()
