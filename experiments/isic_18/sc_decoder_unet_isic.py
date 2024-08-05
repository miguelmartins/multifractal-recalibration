import numpy as np
import os
import tensorflow as tf

from config.custom_utils import get_cuda_device_environ
from etl.preprocessing import get_segmentation_data
from models.recalibrated_unet import get_monofractal_unet, get_se_unet, get_se_everywhere_unet, get_sc_everywhere_unet, \
    get_sc_encoder_unet
from models.unet import get_unet
from config.parser import ExperimentConfigParser
from sklearn.model_selection import KFold

CONFIG_FILE = '/home/miguelmartins/Projects/FractalFCN/config/kvasir-seg-newconf.yaml'
LOG_DIR = '/home/miguelmartins/Projects/FractalFCN/logs_scale_free/five-fold/isic/fix'

NUM_FOLDS = 5


def fix_shape(x, y):
    return tf.transpose(x, perm=[0, 2, 3, 1]), tf.transpose(y, perm=[0, 2, 3, 1])


def normalize_fn(x, y, minimum_value=0., maximum_value=255.):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    x = tf.clip_by_value(x, clip_value_min=minimum_value, clip_value_max=maximum_value)
    y = tf.clip_by_value(y, clip_value_min=minimum_value, clip_value_max=maximum_value)

    return (x - minimum_value) / (maximum_value - minimum_value), (y - minimum_value) / (maximum_value - minimum_value)


NUM_FOLDS = 5


def augment_model():
    flip = tf.keras.layers.RandomFlip(mode="horizontal")
    rotation = tf.keras.layers.RandomFlip(mode="vertical")
    aug_model = tf.keras.Sequential([flip, rotation])
    return aug_model


aug_model = augment_model()


def get_augment_fn(x, y):
    def augment_binary_segmentation(x, y):
        y = tf.cast(y, dtype=tf.float32)
        batch_d = tf.concat([x, y], axis=-1)
        batch_aug = aug_model(batch_d)
        return batch_aug[..., :-1], batch_aug[..., -1:]

    return augment_binary_segmentation(x, y)


def main():
    dataset_np_x = np.load('/home/miguelmartins/Datasets/ISIC2018/np/X_tr_224x224.npy')
    dataset_np_y = np.load('/home/miguelmartins/Datasets/ISIC2018/np/Y_tr_224x224.npy')

    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset_np_x)):
        print(f"Fold {fold} starting...")
        fold_data = ExperimentConfigParser(
            name=f'sc-decoder-unet-{NUM_FOLDS}-fold-{fold}-cuda-{get_cuda_device_environ()}-isic18_pid{os.getpid()}',
            config_path=CONFIG_FILE,
            log_dir=LOG_DIR)
        model = get_sc_encoder_unet(channels_per_level=fold_data.config.model.level_depth,
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
        train = train.map(get_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
