import tensorflow as tf

from neural_components.custom_ops import sample_min_max_scaling

import tensorflow as tf
from typing import Any


class OrdinaryLeastSquares(tf.keras.layers.Layer):

    def __init__(self, max_scale: int, **kwargs: Any) -> None:
        """
        Initializes the OrdinaryLeastSquares layer.

        Args:
            max_scale (int): The maximum scale := 2^{maximum_scale} to be used in the calculation.
            **kwargs: Additional keyword arguments for the layer.
        """
        super(OrdinaryLeastSquares, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The OLS estimate of the slope m:
        m = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)

        where:
        - x_i are the log scales
        - y_i are the log measures
        - mean_x is the mean of log scales
        - mean_y is the mean of log measures
        """
        scales = 2 ** tf.range(1, self.max_scale + 1, dtype=tf.float32)
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)


class WeightedOrdinaryLeastSquares(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(WeightedOrdinaryLeastSquares, self).__init__(**kwargs)
        self.max_scale = max_scale

    def build(self, input_shape):
        self.weights_ols = self.add_weight(name='weights_ols',
                                           shape=[self.max_scale],
                                           initializer=tf.keras.initializers.GlorotUniform(),
                                           dtype=tf.float32,
                                           trainable=True)  # shared weigh
        super(WeightedOrdinaryLeastSquares, self).build(input_shape)

    @tf.function
    def call(self, x):
        scales = 2 ** tf.range(1, self.max_scale + 1,
                               dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = tf.nn.sigmoid(self.weights_ols) * (log_measures - mean_log_measures) * (
                    log_scales - mean_log_scales)
        denominator = tf.nn.sigmoid(self.weights_ols) * (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)


class LocalSingularityStrength(tf.keras.layers.Layer):
    """
    The local singularity strength, also known as the Hölder exponent, is computed by applying a series of
    DepthwiseConv2D operations at different scales and then estimating the local scaling exponent using the OLS method.

    Attributes:
        max_scale (int): The maximum scale to be used in the calculation.
        scales (List[float]): List of scales to be used in the DepthwiseConv2D layers.
        bn (tf.keras.layers.BatchNormalization): Batch normalization layer to normalize the computed exponents.
    """

    def __init__(self, max_scale: int, **kwargs: Any) -> None:
        """
        Args:
            max_scale (int): The maximum scale to be used in the calculation.
            **kwargs: Additional keyword arguments for the layer.
        """
        super(LocalSingularityStrength, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape: Any) -> None:
        """
        Creates the DepthwiseConv2D layers for each scale.
        Args:
            input_shape (Any): Shape of the input tensor.
        """
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(r, r),
                    depth_multiplier=1,
                    trainable=False,
                    activation=None,
                    padding="SAME",
                    depthwise_initializer=tf.keras.initializers.Ones()
                )
            )
        self.scales = tf.cast(self.scales, dtype=tf.float32)
        super(LocalSingularityStrength, self).build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Boolean indicating if the batch norm should operate in training mode.

        Returns:
            tf.Tensor: The estimated local singularity strength (Hölder exponent).
        """
        x = sample_min_max_scaling(x)  # This step can be removed if one ensures that x is non-negative
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        return self.bn(alphas, training=training)


class SharedSoftGroupAssignment(tf.keras.layers.Layer):
    """
    A TensorFlow Layer that determines a (soft) membership function for a set of (feature) point set categorizations
    in a data driven fashion.
    Implementation of Point Grouping Block (equations 9 through 12) of:
    Encoding Spatial Distribution of Convolutional Features for Texture Representation by Yong Xu et al.
    """

    def __init__(self, num_anchors, per_channel=False, **kwargs):
        self.num_anchors = num_anchors
        self.per_channel = per_channel
        self.bn = tf.keras.layers.BatchNormalization()
        super(SharedSoftGroupAssignment, self).__init__(**kwargs)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        # TODO: investigate why these shapes behave similarly
        if self.per_channel:
            self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(num_channels, self.num_anchors),
                                                 dtype=tf.float32,
                                                 initializer="uniform",
                                                 trainable=True)  # TODO: this initialization is different in original implementation

            self.membership_weights = self.add_weight(name="membership_weights", shape=(num_channels, self.num_anchors),
                                                      dtype=tf.float32, initializer="uniform",
                                                      trainable=True
                                                      )
        else:
            self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(1, self.num_anchors),
                                                 dtype=tf.float32,
                                                 initializer="uniform",
                                                 trainable=True)  # TODO: this initialization is different in original implementation

            self.membership_weights = self.add_weight(name="membership_weights", shape=(1, self.num_anchors),
                                                      dtype=tf.float32, initializer="uniform",
                                                      trainable=True
                                                      )

        super(SharedSoftGroupAssignment, self).build(input_shape)

    def call(self, x, training=False):
        anchor_feature_tensor = (x[..., tf.newaxis] - self.anchor_tensor) ** 2
        membership_matrix = -self.membership_weights * anchor_feature_tensor
        soft_histogram = tf.nn.softmax(membership_matrix, axis=-1)
        return self.bn(soft_histogram, training)


class SoftHistogramLayer(tf.keras.layers.Layer):
    """
    Inspired by the implementation of Point Grouping Block, equations 9 through 12 of:
    Encoding Spatial Distribution of Convolutional Features for Texture Representation by Yong Xu et al.

    However, our implementation shares parameters across channels by default.
    """

    def __init__(self, num_anchors, per_channel=False, **kwargs):
        self.num_anchors = num_anchors
        self.per_channel = per_channel
        self.bn = tf.keras.layers.BatchNormalization()
        super(SoftHistogramLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        if self.per_channel:
            self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(num_channels, self.num_anchors),
                                                 dtype=tf.float32,
                                                 initializer="uniform",
                                                 trainable=True)

            self.membership_weights = self.add_weight(name="membership_weights", shape=(num_channels, self.num_anchors),
                                                      dtype=tf.float32, initializer="uniform",
                                                      trainable=True
                                                      )
        else:
            self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(1, self.num_anchors),
                                                 dtype=tf.float32,
                                                 initializer="uniform",
                                                 trainable=True)

            self.membership_weights = self.add_weight(name="membership_weights", shape=(1, self.num_anchors),
                                                      dtype=tf.float32, initializer="uniform",
                                                      trainable=True
                                                      )

        super(SoftHistogramLayer, self).build(input_shape)

    def call(self, x, training=False):
        anchor_feature_tensor = (x[..., tf.newaxis] - self.anchor_tensor) ** 2
        membership_matrix = -tf.nn.relu(self.membership_weights) * anchor_feature_tensor
        soft_histogram = tf.nn.softmax(membership_matrix, axis=-1)
        return self.bn(soft_histogram, training)
