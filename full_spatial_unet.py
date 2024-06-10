import tensorflow as tf
from tensorflow import keras
from keras import layers
import math

from setup import *


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        #x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.LayerNormalization(axis=-1,center=True, scale=True)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def Spatial_attention(x):
    avg_out = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_out = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_out, max_out])
    spatial_attention_feature = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    
    return layers.Multiply()([x, spatial_attention_feature])



def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            
            x = Spatial_attention(x)

            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
            
            x = Spatial_attention(x)
        return x

    return apply



def get_spatial_unet(image_size, input_frames, output_frames, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, input_frames))
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)
    x = layers.Conv2D(1, 3, padding="valid")(x)
    x = layers.Conv2D(1, 2, padding="valid")(x)
    
    return keras.Model([noisy_images], x, name="residual_unet")
