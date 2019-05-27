import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import re
import time


def make_spam_dataset(hparams):
  docs_train, docs_test, labels_train, labels_test = get_spam_info(hparams)
  time.sleep(5)
  train_dataset = tf.data.Dataset.zip(
      (tf.data.Dataset.from_tensor_slices(docs_train),
       tf.data.Dataset.from_tensor_slices(labels_train))).shuffle(
           buffer_size=len(labels_train) * 2).batch(hparams.batch_size).repeat(
               hparams.num_epochs)

  test_dataset = tf.data.Dataset.zip(
      (tf.data.Dataset.from_tensor_slices(docs_test),
       tf.data.Dataset.from_tensor_slices(labels_test))).shuffle(
           buffer_size=len(labels_test) * 2).batch(hparams.batch_size).repeat(
               hparams.num_epochs)

  train_iterator, test_iterator = train_dataset.make_initializable_iterator(
  ), test_dataset.make_initializable_iterator()

  return train_iterator, test_iterator


def clean_str(string):
  """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)

  return string.strip().lower()


def get_spam_info(hparams, return_only_mapping=False):
  docs = []
  labels = []
  for dir in hparams.data_dir:
    for file in os.listdir(dir):
      if file.startswith('t_'):
        labels.append(1)
      else:
        labels.append(0)
      text = ''
      for line in open(os.path.join(dir, file)).readlines():
        text += line
      docs.append(text)

  docs = [clean_str(d) for d in docs]
  hparams.max_length = max([len(x.split(" ")) for x in docs])
  vocab_processor = learn.preprocessing.VocabularyProcessor(hparams.max_length)
  docs = np.asarray(list(vocab_processor.fit_transform(docs)))
  if return_only_mapping:
    return vocab_processor.vocabulary_._reverse_mapping
  hparams.vocabulary_size = np.max(docs) + 1
  X_train, X_test, y_train, y_test = train_test_split(docs,
                                                      labels,
                                                      test_size=0.20,
                                                      random_state=42)

  return X_train, X_test, y_train, y_test
