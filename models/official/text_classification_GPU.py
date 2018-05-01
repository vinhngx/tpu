from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf

FLAGS = None

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
CHARS_FEATURE = 'chars'  # Name of the input character feature.


def char_rnn_model(features, labels, mode):
  """Character level recurrent neural network model to predict classes."""
  byte_vectors = tf.one_hot(features[CHARS_FEATURE], 256, 1., 0.)
  byte_list = tf.unstack(byte_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  #cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
  outputs, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Prepare training and testing data
  dbpedia = tf.contrib.learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=False)
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.DataFrame(dbpedia.test.data)[1]
  y_test = pandas.Series(dbpedia.test.target)

  # Process vocabulary
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(
      MAX_DOCUMENT_LENGTH)
  x_train_fit = np.array(list(char_processor.fit_transform(x_train)))
  x_test_fit = np.array(list(char_processor.transform(x_test)))

  # Build model
  classifier = tf.estimator.Estimator(model_fn=char_rnn_model)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={CHARS_FEATURE: x_train_fit},
      y=y_train,
      batch_size=128,
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=1000)

  # Eval.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={CHARS_FEATURE: x_test_fit},
      y=y_test,
      num_epochs=1,
      shuffle=False)

  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)