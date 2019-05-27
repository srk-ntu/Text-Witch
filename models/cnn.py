import tensorflow as tf
import numpy as np
from .registry import register


@register
class CNN(tf.keras.Model):

  def __init__(self, hparams):
    super().__init__()
    self._hparams = hparams
    self.conv_block_layers = []
    self.W = tf.get_variable(
        "W",
        initializer=tf.random_uniform([self._hparams.vocabulary_size, 128],
                                      -1.0, 1.0))
    for filter_size, num_filters in zip(hparams.filter_sizes,
                                        hparams.num_filters):
      self.conv_block_layers.append(
          (tf.keras.layers.Conv2D(num_filters,
                                  (filter_size, hparams.embedding_dim)),
           tf.keras.layers.MaxPool2D(pool_size=(hparams.max_length -
                                                filter_size + 1, 1),
                                     strides=(1, 1))))
    self.reshape = tf.keras.layers.Reshape((-1, np.sum(hparams.num_filters)))
    self.dropout = tf.keras.layers.Dropout(hparams.drop_rate)
    self.dense = tf.keras.layers.Dense(hparams.num_classes)

  def call(self, x, rate):

    embedded_chars = tf.nn.embedding_lookup(self.W, x)
    embeddings = tf.expand_dims(embedded_chars, axis=-1)
    pooled_outputs = []
    for conv, pool in self.conv_block_layers:
      y = conv(embeddings)
      pooled_outputs.append(pool(y))
    output = tf.concat(pooled_outputs, 3)
    output = self.reshape(output)
    output = tf.nn.dropout(output, rate=rate)

    return self.dense(output)
