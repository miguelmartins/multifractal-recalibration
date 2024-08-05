import tensorflow as tf

from neural_components.attention import MonofractalRecalibration, SqueezeExcite, \
    MultifractalRecalibration, StyleBasedRecalibration, \
    MultiSpectralChannelAttention
from neural_components.convolutional import ContractingLayer, UpsampleExpandingLayer


def get_monofractal_unet(channels_per_level, r=2, max_scale=3, input_shape=[224, 224, 3], n_classes=2, with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = MonofractalRecalibration(r=r, max_scale=max_scale)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])


def get_se_unet(channels_per_level, r=2, max_scale=3, input_shape=[224, 224, 3], n_classes=2, with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = SqueezeExcite(r=r)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])


def get_sc_encoder_unet(channels_per_level, r=2, max_scale=3, input_shape=[224, 224, 3], n_classes=2, with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = SpatialSqueezeExcite(r=r)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])


def get_srm_encoder_unet(channels_per_level, r=2, max_scale=3, input_shape=[224, 224, 3], n_classes=2, with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = StyleBasedRecalibration()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])


def get_fca_encoder_unet(channels_per_level, r=2, low_freq=4, input_shape=[224, 224, 3], n_classes=2, with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = MultiSpectralChannelAttention(r=2, low_freq=low_freq)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])


def get_multifractal_unet(channels_per_level, max_scale=3, k=16, per_channel=False, input_shape=[224, 224, 3],
                          n_classes=2,
                          with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for i, n_channel in enumerate(channels_per_level[:-1]):
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = MultifractalRecalibration(max_scale=max_scale, k=k, per_channel=per_channel)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])
