from pathlib import Path

import numpy as np
import os
import tensorflow as tf

from config.custom_utils import get_cuda_device_environ, get_config_directory
from etl.preprocessing import get_segmentation_data, normalize_fn, AugmentationModel
from models.recalibrated_unet import get_srm_encoder_unet
from config.parser import ExperimentConfigParser
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_DIR = script_dir = Path(__file__).resolve().parent.parent
CONFIG_PATH = get_config_directory()

CONFIG_FILE = os.path.join(CONFIG_PATH, 'files/config.yaml')
LOG_DIR = os.path.join(ROOT_DIR, 'busi/five_fold')
DATA_PATH = '/home/miguelmartins/Datasets/BUSI_sorted'

NUM_FOLDS = 5


def main():
    config_data = ExperimentConfigParser(name=f'busi_pid{os.getpid()}',
                                         config_path=CONFIG_FILE,
                                         log_dir=LOG_DIR)
    dataset = get_segmentation_data(img_path=os.path.join(DATA_PATH, 'images'),
                                    msk_path=os.path.join(DATA_PATH, 'masks'),
                                    batch_size=1,
                                    target_size=config_data.config.data.target_size)
    dataset = dataset.map(normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_np_x = np.array([x for (x, _) in dataset]).squeeze(axis=1)
    dataset_np_y = np.array([y for (_, y) in dataset]).squeeze(axis=1)

    file_dirs = os.listdir(os.path.join(DATA_PATH, 'images'))
    file_dirs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    is_benign = np.array([0 if file.split(' ')[0] == 'benign' else 1 for file in sorted(file_dirs)])

    aug_model = AugmentationModel()
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(X=dataset_np_x, y=is_benign)):
        print(f"Fold {fold} starting...")
        fold_data = ExperimentConfigParser(
            name=f'srm-encoder-unet-cuda{get_cuda_device_environ()}-{NUM_FOLDS}-folds-{fold}-busi_pid{os.getpid()}',
            config_path=CONFIG_FILE,
            log_dir=LOG_DIR)
        model = get_srm_encoder_unet(channels_per_level=fold_data.config.model.level_depth,
                         input_shape=config_data.config.data.target_size + [3],
                         with_bn=False)
        model.compile(loss=fold_data.loss_object,
                      optimizer=fold_data.optimizer_obj,
                      metrics=fold_data.metrics)

        x_fold, y_fold = dataset_np_x[train_index], dataset_np_y[train_index]
        is_benign_fold = is_benign[train_index]
        x_train, x_dev, y_train, y_dev = train_test_split(x_fold, y_fold,
                                                          test_size=0.2,
                                                          stratify=is_benign_fold,
                                                          random_state=42)

        x_val, y_val = dataset_np_x[val_index], dataset_np_y[val_index]

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
