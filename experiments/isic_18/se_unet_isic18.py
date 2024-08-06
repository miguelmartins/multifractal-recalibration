from pathlib import Path

import numpy as np
import os
import tensorflow as tf

from config.custom_utils import get_cuda_device_environ, get_config_directory
from etl.preprocessing import fix_shape, AugmentationModel
from models.recalibrated_unet import get_se_unet
from config.parser import ExperimentConfigParser
from sklearn.model_selection import KFold

ROOT_DIR = script_dir = Path(__file__).resolve().parent.parent
CONFIG_PATH = get_config_directory()

CONFIG_FILE = os.path.join(CONFIG_PATH, 'files/config.yaml')
LOG_DIR = os.path.join(ROOT_DIR, 'isic18/five_fold')

NUM_FOLDS = 5


def main():
    dataset_np_x = np.load('/home/miguelmartins/Datasets/ISIC2018/np/X_tr_224x224.npy')
    dataset_np_y = np.load('/home/miguelmartins/Datasets/ISIC2018/np/Y_tr_224x224.npy')

    aug_model = AugmentationModel()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset_np_x)):
        print(f"Fold {fold} starting...")
        fold_data = ExperimentConfigParser(
            name=f'se-unet-{NUM_FOLDS}-fold-{fold}-cuda-{get_cuda_device_environ()}-isic18_pid{os.getpid()}',
            config_path=CONFIG_FILE,
            log_dir=LOG_DIR)
        model = get_se_unet(channels_per_level=fold_data.config.model.level_depth,
                            input_shape=fold_data.config.data.target_size + [3],
                            with_bn=False)  # RGB
        model.compile(loss=fold_data.loss_object,
                      optimizer=fold_data.optimizer_obj,
                      metrics=fold_data.metrics)
        print(model.summary())
        x_fold, y_fold = dataset_np_x[train_index], dataset_np_y[train_index]
        x_val, y_val = dataset_np_x[val_index], dataset_np_y[val_index]
        dev_size = int(len(x_fold) * 0.1)
        dev_idx = len(x_fold) - dev_size
        x_train, y_train = x_fold[:dev_idx], y_fold[:dev_idx]
        x_dev, y_dev = x_fold[dev_idx:], y_fold[dev_idx:]

        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dev = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
        test = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        train = train.batch(fold_data.config.training.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dev = dev.batch(fold_data.config.training.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        test = test.batch(fold_data.config.training.batch_size, num_parallel_calls=tf.data.AUTOTUNE)

        train = train.map(fix_shape, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.map(aug_model.augment_binary_segmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dev = dev.map(fix_shape, num_parallel_calls=tf.data.AUTOTUNE)
        test = test.map(fix_shape, num_parallel_calls=tf.data.AUTOTUNE)

        model.fit(train,
                  validation_data=dev,
                  epochs=fold_data.config.training.epochs,
                  batch_size=fold_data.config.training.batch_size,
                  callbacks=fold_data.callbacks
                  )
        model.load_weights(fold_data.model_checkpoint_path)
        model.evaluate(test, callbacks=fold_data.test_callbacks)
        fold_data.dump_config(description=f'{fold}')
        print()


if __name__ == '__main__':
    main()
