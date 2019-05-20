from datasets.day_night import DayNightDataset
from models.SimpleModel import SimpleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'path_day': 'data/sun/right',
        'path_night': 'data/night/right',
    },
    model=SimpleModel,
    model_args={},
    batch_size=50,
    epochs=1,
    val_size=0.2,
    log_path='log',
)

trainer = Trainer(config)

trainer.train()
