import tensorflow as tf
from keras import backend as K


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def dice_coefficient(y_true, y_pred, epsilon=1e-12):
    # corrected from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    _y_true = tf.keras.layers.Flatten()(y_true)
    _y_pred = tf.where(tf.keras.layers.Flatten()(y_pred) >= 0.5, 1., 0.)
    intersection = tf.reduce_sum(_y_pred * _y_true, axis=-1)
    return (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, axis=-1) + tf.reduce_sum(_y_pred, axis=-1) + epsilon)


@tf.function
def dice_coefficient_multi(y_true, y_pred, epsilon=1e-12):
    n_classes = y_true.shape[-1] - 1
    dices = 0  # one can initize as 0 since tensors have __plus__ defined having scalars
    for class_ in range(1, n_classes):
        dices += dice_coefficient(y_true[..., class_], y_pred[..., class_], epsilon)
    return dices / n_classes


def get_baseline_segmentation_metrics():
    accuracy = tf.keras.metrics.BinaryAccuracy()
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    sens = tf.keras.metrics.Recall(name='sensitivity')
    spec = tf.keras.metrics.MeanMetricWrapper(specificity, name='specificity')
    dice_score = tf.keras.metrics.MeanMetricWrapper(fn=dice_coefficient, name='dice_score')
    iou = tf.keras.metrics.BinaryIoU(target_class_ids=[1])
    return [accuracy, auc, prec, sens, spec, dice_score, iou]


def get_baseline_segmentation_metrics_multi():
    return [tf.keras.metrics.MeanMetricWrapper(fn=dice_coefficient_multi, name='mean_dice_score')]
