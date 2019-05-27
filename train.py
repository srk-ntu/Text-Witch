import tensorflow as tf
import numpy as np
import os

from models.registry import get_model
from hparams.registry import get_hparams
from data.load_data import make_spam_dataset

tf.flags.DEFINE_string("hparams", None, "Hyperparameters")
tf.flags.DEFINE_string("output_dir", None, "where to store")

FLAGS = tf.app.flags.FLAGS

hparams = get_hparams(FLAGS.hparams)
train_iterator, test_iterator = make_spam_dataset(hparams)
model = get_model(hparams.model)(hparams)

model_save_path = os.path.join(FLAGS.output_dir, 'model')
train_summary_dir = os.path.join(FLAGS.output_dir, "summaries", "train")
test_summary_dir = os.path.join(FLAGS.output_dir, "summaries", "test")


def build():
  x = tf.placeholder(tf.int32, shape=[None, hparams.max_length], name="input")
  y = tf.placeholder(tf.int64, shape=[None], name="labels")
  drop_rate = tf.placeholder_with_default(0.0, shape=())

  logits = model(x, drop_rate)
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=tf.one_hot(y,
                                                                   depth=2,
                                                                   axis=-1))
  loss = tf.reduce_mean(loss)
  train_op = tf.train.AdamOptimizer(hparams.lr).minimize(
      loss, var_list=model.trainable_weights)
  preds = tf.squeeze(tf.argmax(logits, axis=-1), axis=1)
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(y, preds)))

  def create_summaries():
    loss_summary = tf.summary.scalar("Loss: ", loss)
    accuracy_summary = tf.summary.scalar("Train accuracy", accuracy)
    train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                 graph=tf.get_default_graph())
    test_summary_writer = tf.summary.FileWriter(test_summary_dir,
                                                graph=tf.get_default_graph())

    return train_summary_writer, test_summary_writer, train_summary_op, test_summary_op

  def data_initializers():
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)

  def initialize_vars():
    sess.run(
        tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))

  def test(step):
    for i in range(hparams.eval_steps):
      inputs_test, labels_test = sess.run([test_images, test_labels])
      accuracy_test, summary = sess.run([accuracy, test_sop],
                                        feed_dict={
                                            x: inputs_test,
                                            y: labels_test,
                                            drop_rate: 0.0
                                        })

      print("Test Accuracy: ", accuracy_test)
      test_writer.add_summary(summary, step + i)

  def train():
    for step in range(start_step, 50 * hparams.num_epochs):
      inputs, labels = sess.run([input_t, label_t])
      _, loss_, accuracy_, train_summary_ = sess.run(
          [train_op, loss, accuracy, train_sop],
          feed_dict={
              x: inputs,
              y: labels,
              drop_rate: hparams.drop_rate
          })

      train_writer.add_summary(train_summary_, step)

      if step % hparams.save_every == 0:
        saver.save(sess, model_save_path + '.ckpt', global_step=step)

      if step % hparams.test_every == 0:
        test(step)
      print("Step: {}  Loss: {}  Accuracy: {}".format(step, loss_, accuracy_))

  start_step = 0
  train_writer, test_writer, train_sop, test_sop = create_summaries()
  sess = tf.Session()
  saver = tf.train.Saver(max_to_keep=10)
  if os.path.isfile(os.path.join(FLAGS.output_dir, 'checkpoint')):
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.output_dir)
    start_step = int(latest_ckpt.split('-')[-1])
    saver.restore(sess, latest_ckpt)
  else:
    initialize_vars()

  data_initializers()

  input_t, label_t = train_iterator.get_next()
  test_images, test_labels = test_iterator.get_next()
  train()
  train_writer.close()
  test_writer.close()


def main(_):
  build()


if __name__ == '__main__':
  tf.app.run()