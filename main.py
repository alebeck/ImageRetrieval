from datasets.day_night import DayNightDataset
from models.simple_model import SimpleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'paths_day': [
            'data/sun/right',
            'data/sun/left',
        ],
        'paths_night': [
            'data/night/right',
            'data/night/left',
        ],
    },
    model=SimpleModel,
    model_args={},
    batch_size=64,
    epochs=10,
    val_size=0.2,
    log_path='log',
    save_every=100
)

trainer = Trainer(config)

print(f'{len(trainer.data)} batches')

trainer.train()
