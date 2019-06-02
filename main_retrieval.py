from datasets.matched_images import MatchedImagesDataset
from datasets.day_night import DayNightDataset
from models.simple_model import SimpleModel
from utils.config import RetrievalConfig, TrainingConfig
from utils import retrieval
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=DayNightDataset,
    dataset_args={
        'paths_day': [
            'data/sun/right',
            'data/sun/left',
            'data/overcast-summer/right',
            'data/overcast-summer/left',
        ],
        'paths_night': [
            'data/night/right',
            'data/night/left',
            'data/night-rain/right',
            'data/night-rain/left',
        ],
    },
    model=SimpleModel,
    model_args={},
    batch_size=64,
    epochs=0,
    val_size=0.2,
    log_path='log',
    save_every=10000
)

trainer = Trainer(config)

print(f'{len(trainer.data.train_loader)} batches')

trainer.train()


####################################################


encoder_day = trainer.model.ae_day
encoder_night = trainer.model.ae_night


config_day2night = RetrievalConfig(
    dataset=MatchedImagesDataset,
    dataset_args={
        'paths_anchors': ['data/pairs/day'],
        'paths_opposites': ['data/pairs/night'],
    },
    batch_size=64,
)

config_night2day = RetrievalConfig(
    dataset=MatchedImagesDataset,
    dataset_args={
        'paths_anchors': ['data/pairs/night'],
        'paths_opposites': ['data/pairs/day'],
    },
    batch_size=64,
)

model = 0

loss_day2night = retrieval.evaluate(encoder_day, encoder_night, config_day2night)
print(f'Loss day to night: {loss_day2night}')
loss_night2day = retrieval.evaluate(encoder_night, encoder_day, config_night2day)
print(f'Loss night to day: {loss_night2day}')
