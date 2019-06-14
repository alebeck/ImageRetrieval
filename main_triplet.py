from datasets.embedding import EmbeddingDataset
from models.feature_weight import FeatureWeight
from models.simple_model import SimpleModel
from utils.config import TrainingConfig
from utils.trainer import Trainer


config = TrainingConfig(
    dataset=EmbeddingDataset,
    dataset_args={
        'model_class': SimpleModel,
        'model_args': {},
        'weights_path': '../weights/1950_weights.pt',
        'paths_day': [
            '../data/pairs/day'
        ],
        'paths_night': [
            '../data/pairs/night'
        ]
    },
    model=FeatureWeight,
    model_args={ 'layers': {'conv1': 8, 'conv2': 16} },
    batch_size=2,
    epochs=10,
    val_size=0.2,
    log_path='log_triplet',
    save_every=100
)

Trainer(config).train()