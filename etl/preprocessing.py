import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import GroupKFold
from tqdm import tqdm


def get_segmentation_data(*, img_path, msk_path, batch_size, target_size):
    image_gen = tf.keras.utils.image_dataset_from_directory(
        img_path,
        labels=None,
        label_mode=None,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        image_size=target_size,
        subset=None,
        interpolation='bilinear',
    )

    mask_gen = tf.keras.utils.image_dataset_from_directory(
        msk_path,
        labels=None,
        label_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False,
        image_size=target_size,
        subset=None,
        interpolation='nearest',
    )

    return tf.data.Dataset.zip((image_gen, mask_gen))


class SynapseDataPreparer:
    def __init__(self, data_path, n_folds=5, n_classes=9):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train')
        self.test_path = os.path.join(data_path, 'test')
        self.n_folds = n_folds
        self.n_classes = n_classes
        self.group_k_fold = GroupKFold(n_splits=self.n_folds)
        self.train_groups = self._get_train_groups()

    def _get_train_groups(self):
        files = os.listdir(self.train_path)
        # check case number id (string structure: caseid_slinenumer.npz)
        cases = [file.split('_')[0] for file in files]
        # map each case to unique index
        mapping = {case_number: index for index, case_number in enumerate(sorted(set(cases)))}
        # apply mapping for all files listed in directory
        groups = [mapping[case] for case in cases]
        return np.array(groups)

    def get_next_split(self):
        file_dirs = np.array([os.path.join(self.train_path, file) for file in os.listdir(self.train_path)])
        slice_splits = self.group_k_fold.split(X=file_dirs, y=None, groups=self.train_groups)

        for train_idx, val_idx in tqdm(slice_splits, total=self.n_folds):
            train_slices, val_slices = file_dirs[train_idx], file_dirs[val_idx]
            train_x, train_y = [], []
            val_x, val_y = [], []

            for slice_train in train_slices:
                slice_train_np = np.load(slice_train)
                train_x.append(slice_train_np['image'])
                train_y.append(slice_train_np['label'])
                slice_train_np.close()

            for slice_val in val_slices:
                slice_val_np = np.load(slice_val)
                val_x.append(slice_val_np['image'])
                val_y.append(slice_val_np['label'])
                slice_val_np.close()

            yield np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)


class AugmentationModel:
    def __init__(self):
        self.aug_model = self._create_augment_model()

    def _create_augment_model(self):
        flip = tf.keras.layers.RandomFlip(mode="horizontal")
        rotation = tf.keras.layers.RandomFlip(mode="vertical")
        return tf.keras.Sequential([flip, rotation])

    def augment_binary_segmentation(self, x, y):
        y = tf.cast(y, dtype=tf.float32)
        batch_d = tf.concat([x, y], axis=-1)
        batch_aug = self.aug_model(batch_d)
        return batch_aug[..., :-1], batch_aug[..., -1:]


def normalize_fn(x, y, minimum_value=0., maximum_value=255.):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    x = tf.clip_by_value(x, clip_value_min=minimum_value, clip_value_max=maximum_value)
    y = tf.clip_by_value(y, clip_value_min=minimum_value, clip_value_max=maximum_value)

    return (x - minimum_value) / (maximum_value - minimum_value), (y - minimum_value) / (maximum_value - minimum_value)


def fix_shape(x, y):
    return tf.transpose(x, perm=[0, 2, 3, 1]), tf.transpose(y, perm=[0, 2, 3, 1])
