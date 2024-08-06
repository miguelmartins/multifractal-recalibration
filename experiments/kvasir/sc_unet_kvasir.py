from pathlib import Path

import numpy as np
import os
import tensorflow as tf

from config.custom_utils import get_cuda_device_environ, get_config_directory
from etl.preprocessing import get_segmentation_data, normalize_fn, AugmentationModel
from models.recalibrated_unet import get_sc_encoder_unet
from config.parser import ExperimentConfigParser
from sklearn.model_selection import KFold

ROOT_DIR = script_dir = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = get_config_directory()

CONFIG_FILE = os.path.join(CONFIG_PATH, 'files/config.yaml')
LOG_DIR = os.path.join(ROOT_DIR, 'kvasir/five_fold')
DATA_PATH = '/home/miguelmartins/Datasets/kvasir-seg/Kvasir-SEG/'

NUM_FOLDS = 5

def main():
    config_data = ExperimentConfigParser(name=f'kvasir-seg_pid{os.getpid()}',
                                         config_path=CONFIG_FILE,
                                         log_dir=LOG_DIR)
    dataset = get_segmentation_data(img_path=os.path.join(DATA_PATH, 'images'),
                                    msk_path=os.path.join(DATA_PATH, 'masks'),
                                    batch_size=1,
                                    target_size=config_data.config.data.target_size)
    dataset = dataset.map(normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_np_x = np.array([x for (x, _) in dataset]).squeeze(axis=1)
    dataset_np_y = np.array([y for (_, y) in dataset]).squeeze(axis=1)

    aug_model = AugmentationModel()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset_np_x)):
        print(f"Fold {fold} starting...")
        fold_data = ExperimentConfigParser(
            name=f'sc-unet-cuda{get_cuda_device_environ()}-{NUM_FOLDS}-folds-{fold}-kvasir-seg_pid{os.getpid()}',
            config_path=CONFIG_FILE,
            log_dir=LOG_DIR)
        model = get_sc_encoder_unet(channels_per_level=fold_data.config.model.level_depth,
                                    input_shape=config_data.config.data.target_size + [3],
                                    with_bn=False)
        model.compile(loss=fold_data.loss_object,
                      optimizer=fold_data.optimizer_obj,
                      metrics=fold_data.metrics)
        
        x_fold, y_fold = dataset_np_x[train_index], dataset_np_y[train_index]
        x_val, y_val = dataset_np_x[val_index], dataset_np_y[val_index]
        dev_size = int(len(x_fold) * 0.1)
        dev_idx = len(x_fold) - dev_size
        x_train, y_train = x_fold[:dev_idx], y_fold[:dev_idx]
        x_dev, y_dev = x_fold[dev_idx:], y_fold[dev_idx:]

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(fold_data.config.training.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(aug_model.augment_binary_segmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        model.fit(train_ds,
                  epochs=fold_data.config.training.epochs,
                  batch_size=fold_data.config.training.batch_size,
                  validation_data=(x_dev, y_dev),
                  callbacks=fold_data.callbacks,
                  shuffle=True
                  )
        model.load_weights(fold_data.model_checkpoint_path)
        model.evaluate(x=x_val, y=y_val, callbacks=fold_data.test_callbacks)
        fold_data.dump_config(description=f'fold {fold}')


if __name__ == '__main__':
    main()
