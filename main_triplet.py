from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor

from datasets.embedding import EmbeddingDataset
from models.feature_weight import FeatureWeight
from models.simple_model import SimpleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer

# DFM layers and number of channels per layer
layers = {
    'relu3_1': 256,
    'relu3_2': 512,
    'relu3_3': 512,
    'pool3': 512
}

config = TrainingConfig(
    dataset=EmbeddingDataset,
    dataset_args={
        'model_class': SimpleModel,
        'model_args': {},
        'layers': layers,
        'weights_path': '../weights/1950_weights.pt',
        'paths_day': [
            '../data/pairs/day'
        ],
        'paths_night': [
            '../data/pairs/night'
        ],
        'transform': Compose([
            CenterCrop(760),
            Resize(128),
            ToTensor()
        ])
    },
    model=FeatureWeight,
    model_args={ 'layers': layers },
    checkpoint_path=None,
    batch_size=16,
    epochs=10000,
    val_size=0.2,
    log_path='log_triplet',
    save_every=100
)

Trainer(config).train()