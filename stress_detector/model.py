from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input, Flatten, InputLayer, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

import contextlib
@contextlib.contextmanager # TODO
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

def build_cnn(
    num_signals: int = 6,
    num_output_class: int = 2,
    dropout_rate: float = 0.3
) -> tf.keras.models.Sequential:
    with options({"layout_optimizer": False}): #TODO
        model = tf.keras.models.Sequential([
            # input_shape = 14 Signale (bei uns max. 6) X 210 Inputs (aus Tabelle nach Fourier)
            InputLayer(input_shape=(num_signals, 210, 1)),
            Conv2D(filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"),
            Dropout(rate=dropout_rate),
            Conv2D(filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"),
            Dropout(rate=dropout_rate),
            Conv2D(filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"),
            MaxPooling2D(pool_size=(1, 2)),
            Dropout(rate=dropout_rate),
            Conv2D(filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"),
            Dropout(rate=dropout_rate),
            MaxPooling2D(pool_size=(1, 2)),
            Dropout(rate=dropout_rate),
            Flatten(),
            Dense(units=128, activation="relu", kernel_initializer="glorot_uniform"),
            Dropout(rate=dropout_rate),
            Dense(units=64, activation="relu", kernel_initializer="glorot_uniform"),
            Dropout(rate=dropout_rate),
            # Anzahl der Output Units = Anzahl der Klassen (2 - non-stress vs stress)
            # sigmoid statt softmax, da nur 2 Klassen
            Dense(units=num_output_class, activation="sigmoid")
        ])
    
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer(
    num_signals: int = 6,
    num_output_class: int = 2,
    head_size: int = 256,
    num_heads: int = 4,
    ff_dim: int = 4,
    num_transformer_blocks: int = 8,
    mlp_units: list = [128],
    dropout: float = 0.25,
    mlp_dropout: float = 0.25
) -> tf.keras.Model:
    inputs = Input((num_signals, 210))
    x = inputs
    #layers.LSTM(1)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_output_class, activation="sigmoid")(x)
    
    return tf.keras.Model(inputs, outputs)