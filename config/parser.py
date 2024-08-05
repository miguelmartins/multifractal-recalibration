import datetime
from typing import List
import os
import yaml
import tensorflow as tf

from dataclasses import asdict
from evaluation.metrics import get_baseline_segmentation_metrics, get_baseline_segmentation_metrics_multi
from optimization.loss_functions import dice_loss, weighted_loss, dice_loss_multi
from config.custom_dataclasses import ModelCheckpointConfig, ExperimentConfig, CallbacksConfig, DataConfig, \
    TrainingConfig, PlateauSchedulerConfig, ModelConfig


class ExperimentConfigParser:
    def __init__(self, *, name: str, config_path: str, log_dir: str):
        self.name = name
        self.config_path = config_path

        self.timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        self.config = self._get_configs(self.config_path)
        self.log_path = os.path.join(os.path.join(log_dir, self.name),
                                     self.timestamp)
        self.test_log_path = os.path.join(self.log_path, 'test')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.test_log_path, exist_ok=True)
        self._parse_config()

    @staticmethod
    def _get_configs(config_path: str):
        with open(config_path, 'r') as file:
            yaml_config = yaml.safe_load(file)
        # Create a dataclass instance from the loaded YAML
        config = ExperimentConfig(
            callbacks=CallbacksConfig(
                model_checkpoint=ModelCheckpointConfig(**yaml_config['callbacks']['model_checkpoint']),
                scheduler=PlateauSchedulerConfig(**yaml_config['callbacks']['scheduler'])
            ),
            data=DataConfig(**yaml_config['data']),
            training=TrainingConfig(**yaml_config['training']),
            model=ModelConfig(**yaml_config['model'])

        )

        return config

    def _parse_loss(self):
        loss_ = self.config.training.loss
        if loss_ == 'ce':
            return tf.keras.losses.BinaryCrossentropy()
        elif loss_ == 'dice':
            return dice_loss()
        elif loss_ == 'weighted':
            loss_ce = tf.keras.losses.BinaryCrossentropy()
            loss_dice = dice_loss()
            return weighted_loss(0.5, loss_ce, loss_dice)
        elif loss_ == 'weighted_multi':
            loss_ce = tf.keras.losses.CategoricalCrossentropy()
            loss_dice = dice_loss_multi()
            return weighted_loss(0.5, loss_ce, loss_dice)
        else:
            raise ValueError(f"Invalid loss type. Choose from: {', '.join(['ce', 'dice', 'weighted', 'weighted_multi'])}")

    def _parse_metrics(self):
        metrics_ = self.config.training.metrics
        # take advantage of the dataclass to feed into constructor
        if metrics_ == 'baseline':
            return get_baseline_segmentation_metrics()
        elif metrics_ == 'baseline_multi':
            return get_baseline_segmentation_metrics_multi()
        else:
            raise ValueError(f"Invalid metrics type. Choose from{', '.join(['baseline'])}")

    def _parse_optimizer(self):
        optimizer_ = self.config.training.optimizer
        if optimizer_ == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        elif optimizer_ == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.config.training.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer type. Choose from{', '.join(['adam', 'adam'])}")

    def _parse_config(self): # TODO: set train x, y and val x,y and test x,y
        self.loss_object = self._parse_loss()
        self.optimizer_obj = self._parse_optimizer()
        self.metrics = self._parse_metrics()

        self.model_checkpoint_path = os.path.join(self.log_path, 'weights')
        csv_logs_path_ = os.path.join(self.log_path, 'logs.csv')
        mdl_checkpoint_config_ = self.config.callbacks.model_checkpoint.__dict__.copy()
        mdl_checkpoint_config_.update({'filepath': self.model_checkpoint_path})

        scheduler_config_ = self.config.callbacks.scheduler.__dict__.copy()
        scheduler_config_.update({'min_lr': self.config.training.learning_rate})

        self.callbacks = [tf.keras.callbacks.ModelCheckpoint(**mdl_checkpoint_config_),
                          tf.keras.callbacks.CSVLogger(csv_logs_path_, append=True, separator=';'),
                          tf.keras.callbacks.TensorBoard(log_dir=self.log_path),
                          tf.keras.callbacks.ReduceLROnPlateau(**scheduler_config_)]
        self.test_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.test_log_path)]

    def dump_config(self, description=''):
        dict_ = asdict(self.config)
        dict_.update({'description': description})
        dict_.update({'name': self.name})

        with open(os.path.join(self.log_path, 'config.yaml'), 'w') as file:
            yaml.dump(dict_, file)


def str_argument_validator(arg: str, options: List[str]):
    if arg not in options:
        raise ValueError(f"Invalid loss type. Choose from: {', '.join(options)}")
    return arg
