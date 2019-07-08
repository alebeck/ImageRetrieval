from dataclasses import dataclass
from typing import Type, Optional

from torch.utils.data import Dataset
from models.abstract import CustomModule


@dataclass
class TrainingConfig:
    dataset: Type[Dataset]
    dataset_args: dict
    model: Type[CustomModule]
    model_args: dict
    checkpoint_path: Optional[str] # None if not resuming training
    batch_size: int
    epochs: int
    val_size: float
    log_path: str
    save_every: int  # save checkpoint every save_every epochs


@dataclass
class RetrievalConfig:
    dataset: Type[Dataset]
    dataset_args: dict
    batch_size: int
