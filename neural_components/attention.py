from typing import Any

import tensorflow as tf
import numpy as np

from neural_components.custom_ops import sample_min_max_scaling
from neural_components.fractal_geometry import LocalSingularityStrength, SoftHistogramLayer, \
    WeightedOrdinaryLeastSquares


class MultifractalRecalibration(tf.keras.layers.Layer):
    """
    A Keras Layer that performs multifractal recalibration by computing local Hölder exponents
    and approximating their probability using a non-linear mixture of K Gaussians learned as a softmax layer.

    Attributes:
        alpha_layer A layer that computes local Hölder exponents.
        soft_histogram: A layer that approximates the probability of each exponent
                        represented jointly by all channels of the conv. layer
                        using a softmax layer as a surrogate of a non-linear mixture of K Gaussian mixtures.
    """

    def __init__(self, max_scale: Any, k: int, per_channel: bool, **kwargs: Any):
        """
        Args:
            max_scale (Any): Maximum scale parameter for OLS estimate of local singularity strength layer.
            k (int): Number of Gaussian mixtures used in the soft histogram layer.
            per_channel (bool): Whether to compute histograms per channel.
            **kwargs (Any): Additional keyword arguments for the Keras Layer base class.
        """
        super(MultifractalRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape: tf.TensorShape):
        super(MultifractalRecalibration, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying multifractal recalibration.
        """
        alphas = self.alpha_layer(x)
        # Reduce over the K level sets
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)), axis=-1)
        return x + tf.nn.sigmoid(soft_counts)


class WeightedLocalSingularityStrength(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(WeightedLocalSingularityStrength, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()
        self.weighted_ols = WeightedOrdinaryLeastSquares(max_scale=max_scale)

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)
        super(WeightedLocalSingularityStrength, self).build(input_shape)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)  # this step can be removed if one ensures that x is non-negative
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = self.weighted_ols(measures)
        return self.bn(alphas, training=training)


class MonofractalRecalibration(tf.keras.layers.Layer):
    """
    Computes the local scaling exponents for each channel and then
    performs gap followed by an MLP akin to squeeze-and-excite (SE).

    Attributes:
        r (int): Reduction ratio for the intermediate dense layer.
        alpha_layer (LocalSingularityStrength): A layer that computes local singularity strengths.
        gap (tf.keras.layers.GlobalAvgPool2D): Global average pooling layer.
        w1 (tf.keras.layers.Dense): First SE layer for intermediate representation .
        w2 (tf.keras.layers.Dense): Second SE layer for output scaling factors.
    """

    def __init__(self, r: int, max_scale: int, **kwargs: Any):
        """
        Initializes the SingularityStrengthRecalibration layer.

        Args:
            r (int): Reduction ratio for the intermediate dense layer.
            max_scale (Any): Maximum scale parameter for the local singularity strength layer.
            **kwargs (Any): Additional keyword arguments for the Keras Layer base class.
        """
        super(MonofractalRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape: tf.TensorShape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r, activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels, activation='sigmoid')
        super(MonofractalRecalibration, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Performs the forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying singularity strength recalibration.
        """
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]


import tensorflow as tf
from typing import Any, Dict


class SqueezeExcite(tf.keras.layers.Layer):
    """
    Squeeze and Excitation block for channel-wise attention.

    This layer implements the squeeze and excitation operation as described in:
    "Squeeze-and-Excitation Networks" by Jie Hu, Li Shen, and Gang Sun.

    Args:
        r (int): Reduction ratio for the bottleneck layer.
    """

    def __init__(self, r: int, **kwargs: Any):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape: tf.TensorShape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SqueezeExcite, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        squeeze = self.gap(x)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]

    def get_config(self) -> Dict[str, Any]:
        config = super(SqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config


import tensorflow as tf
from typing import Any, Dict


class SpatialSqueezeExcite(tf.keras.layers.Layer):
    """
    scSE layer as in Roy, Abhijit Guha, Nassir Navab, and Christian Wachinger.
    "Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks."

    Args:
        r (int): Reduction ratio for the bottleneck layer.
        **kwargs (Any): Additional keyword arguments for the base Layer class.
    """

    def __init__(self, r: int, **kwargs: Any) -> None:
        super(SpatialSqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape: tf.TensorShape) -> None:
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.conv = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=(1, 1),
                                           activation='sigmoid')
        super(SpatialSqueezeExcite, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        squeeze = self.gap(x)
        channel_excite = x * self.w2(self.w1(squeeze))[:, tf.newaxis, tf.newaxis, :]
        spatial_excite = x * self.conv(x)
        return tf.reduce_max(tf.stack([channel_excite, spatial_excite], axis=-1), axis=-1)

    def get_config(self) -> Dict[str, Any]:
        config = super(SpatialSqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config


class StyleBasedRecalibration(tf.keras.layers.Layer):
    """
    Style-Based Recalibration block for feature recalibration based on global statistics.

    This layer recalibrates the input features based on global average and standard deviation
    statistics, leveraging batch normalization for style-based adjustments.

    Implementation of the method described in Lee, HyunJae, Hyo-Eun Kim, and Hyeonseob Nam.
    "Srm: A style-based recalibration module for convolutional neural networks."
    Proceedings of the IEEE/CVF International conference on computer vision. 2019.

    Args:
        **kwargs (Any): Additional keyword arguments for the base Layer class.
    """

    def __init__(self, **kwargs: Any):
        super(StyleBasedRecalibration, self).__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer weights based on the input shape.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w = tf.keras.layers.Dense(1, use_bias=False, activation=None)
        super(StyleBasedRecalibration, self).build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the layer.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Flag to indicate if the layer is in training mode.

        Returns:
            tf.Tensor: Output tensor after applying style-based recalibration.
        """
        avg = self.gap(x)
        std = tf.math.sqrt(tf.math.reduce_variance(x, axis=[1, 2]) + 1e-12)
        t = tf.keras.layers.Concatenate(axis=-1)([avg[..., tf.newaxis], std[..., tf.newaxis]])
        z = tf.squeeze(self.w(t), axis=-1)
        return x * tf.nn.sigmoid(self.bn(z, training=training))[:, tf.newaxis, tf.newaxis, :]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer.

        Returns:
            Dict[str, Any]: Configuration dictionary for the layer.
        """
        config = super(StyleBasedRecalibration, self).get_config()
        return config


class FrequencyChannelAttention(tf.keras.layers.Layer):
    """
    Multi-Spectral Channel Attention block for feature recalibration using Discrete Cosine Transform (DCT) bases.

    Implementation of the method described in Qin, Zequn, et al.
    "Fcanet: Frequency channel attention networks."
    Proceedings of the IEEE/CVF international conference on computer vision. 2021.

    Args:
        r (int): Reduction ratio for the bottleneck layer. Default is 2.
        low_freq (int): The lower frequency to consider for DCT bases. Default is 4.
    """

    def __init__(self, r: int = 2, low_freq: int = 4, **kwargs: Any):
        super(FrequencyChannelAttention, self).__init__(**kwargs)
        self.r = r
        self.low_freq = low_freq
        self.num_bases = tf.cast(low_freq ** 2, tf.int32)
        self.flatten = tf.keras.layers.Flatten()

    @staticmethod
    def _get_dct_bases(spatial_resolution: int = 224, channel_resolution: int = 32, low_freq: int = None) -> tf.Tensor:
        """
        Generates DCT (Discrete Cosine Transform) bases for given spatial and channel resolutions.
        Args:
            spatial_resolution (int): Spatial resolution of the input tensor. Default is 224.
            channel_resolution (int): Channel resolution of the input tensor. Default is 32.
            low_freq (int): The lower frequency to consider for DCT bases.

        Returns:
            tf.Tensor: DCT bases tensor.
        """
        x = tf.range(spatial_resolution)
        y = tf.range(channel_resolution)
        h, w = tf.meshgrid(x, y, indexing='ij')
        basis_support = tf.cast(tf.stack([h, w], axis=-1), dtype=tf.float32)
        pi = tf.cast(np.pi, dtype=tf.float32)
        left_cos = tf.math.cos((pi * basis_support[..., 0] / spatial_resolution) * (basis_support[..., 1] + 0.5))
        right_cos = tf.math.cos((pi * basis_support[..., 0] / spatial_resolution) * (basis_support[..., 1] + 0.5))
        dct_bases = np.zeros([spatial_resolution, spatial_resolution, channel_resolution, channel_resolution])
        for i in range(spatial_resolution):
            for j in range(spatial_resolution):
                dct_bases[i, j] = tf.linalg.matmul(left_cos[i, ...][..., tf.newaxis],
                                                   right_cos[j, ...][tf.newaxis, ...])
        dct_bases = dct_bases[:, :, :low_freq, :low_freq]
        return tf.cast(dct_bases[np.newaxis, ...],
                       dtype=tf.float32)  # add batch dimension and cast to tf.Tensor float32

    def build(self, input_shape: tf.TensorShape):
        num_channels = input_shape[-1]
        _dct_bases = self._get_dct_bases(spatial_resolution=input_shape[1],
                                         channel_resolution=num_channels,
                                         low_freq=self.low_freq)
        self.dct_bases = tf.reshape(_dct_bases, [_dct_bases.shape[0], _dct_bases.shape[1], _dct_bases.shape[2],
                                                 _dct_bases.shape[3] * _dct_bases.shape[4], 1])

        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(FrequencyChannelAttention, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        num_channels = tf.shape(x)[3]
        x_ = tf.reshape(x, [batch_size, height, width, self.num_bases, num_channels // self.num_bases])
        _freqs = tf.reduce_sum(x_ * self.dct_bases, axis=[1, 2])
        freqs = self.flatten(_freqs)

        return x * self.w2(self.w1(freqs))[:, tf.newaxis, tf.newaxis, :]

    def get_config(self) -> Dict[str, Any]:
        config = super(FrequencyChannelAttention, self).get_config()
        config.update({'r': self.r, 'low_freq': self.low_freq})
        return config
