from dataclasses import dataclass
from typing import List


@dataclass
class ModelCheckpointConfig:
    monitor: str
    save_best_only: bool
    save_weights_only: bool
    patience: int
    mode: str


@dataclass
class PlateauSchedulerConfig:
    monitor: str
    factor: float
    patience: int


@dataclass
class CallbacksConfig:
    model_checkpoint: ModelCheckpointConfig
    scheduler: PlateauSchedulerConfig


@dataclass
class DataConfig:
    target_size: List[int]
    n_classes: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    optimizer: str
    metrics: str


@dataclass
class ModelConfig:
    level_depth: List[int]


@dataclass
class ExperimentConfig:
    callbacks: CallbacksConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
