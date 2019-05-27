import tensorflow as tf
import numpy as np
import os

from models.registry import get_model
from hparams.registry import get_hparams
from data.load_data import make_spam_dataset, get_spam_info

tf.flags.DEFINE_string("hparams", None, "Hyperparameters")
tf.flags.DEFINE_string("output_dir", None, "where to store")
tf.flags.DEFINE_integer("num_words", None, "Number of words to be replaced")

FLAGS = tf.app.flags.FLAGS

hparams = get_hparams(FLAGS.hparams)
train_iterator, test_iterator = make_spam_dataset(hparams)
model = get_model(hparams.model)(hparams)

text, labels = train_iterator.get_next()

model_save_path = os.path.join(FLAGS.output_dir, 'model')
train_summary_dir = os.path.join(FLAGS.output_dir, "summaries", "train")
test_summary_dir = os.path.join(FLAGS.output_dir, "summaries", "test")

word_list = get_spam_info(hparams, True)


def initialize(sess):
  sess.run(train_iterator.initializer)
  sess.run(test_iterator.initializer)


def restore(sess):
  saver = tf.train.Saver(max_to_keep=10)
  latest_ckpt = tf.train.latest_checkpoint(FLAGS.output_dir)
  start_step = int(latest_ckpt.split('-')[-1])
  saver.restore(sess, latest_ckpt)


def get_nearest_embedding(weights, idx):
  original_word = weights[idx]
  distance = np.sum((original_word - weights)**2, axis=1)
  distance = np.sqrt(distance)
  #masking the OG index
  distance[idx] = distance.max()
  nearest_embedding = weights[np.argmin(distance)]

  return np.argmin(distance)


def print_str(array):
  stop_idx = np.where(array == 0)[0][0]
  print(stop_idx)
  strings = [word_list[array[i]] for i in range(0, stop_idx)]

  print(' '.join(strings))


def build():
  x = tf.placeholder(tf.int32, shape=[None, hparams.max_length])
  y = tf.placeholder(tf.int64, shape=[None], name="labels")
  drop_rate = tf.placeholder_with_default(0.0, shape=())

  logits = model(x, drop_rate)
  probs = tf.nn.softmax(logits, axis=-1)

  sess = tf.Session()
  initialize(sess)
  restore(sess)
  embedding_weights = sess.run(model.W)

  inputs, targets = sess.run([text, labels])

  test_ips, test_targs = inputs[0], targets[0]
  print_str(test_ips)

  probabilities = sess.run(probs,
                           feed_dict={
                               x: test_ips[None, :],
                               drop_rate: 0.0
                           })
  print(probabilities)

  indices = np.where(test_ips == 0)[0][0]
  idxes = np.random.choice(np.arange(0, indices), FLAGS.num_words)
  for idx in idxes:
    test_ips[idx] = get_nearest_embedding(embedding_weights, test_ips[idx])
  print("Changed at index: ", idxes)

  print_str(test_ips)
  probabilities = sess.run(probs,
                           feed_dict={
                               x: test_ips[None, :],
                               drop_rate: 0.0
                           })
  print(probabilities)


if __name__ == '__main__':
  build()