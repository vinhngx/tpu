# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:11:13 2018

@author: vinhn
"""

#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of recurrent neural networks over characters for DBpedia dataset.
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626
and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator


FLAGS = None

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
CHARS_FEATURE = 'chars'  # Name of the input character feature.

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'use_tpu', True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

flags.DEFINE_string(
    'master', default=None,
    help='gRPC URL of the master (i.e. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

flags.DEFINE_bool(
    'test_with_fake_data', default=False,
    help='use fake test data')

def char_rnn_model(features, labels, mode):
  """Character level recurrent neural network model to predict classes."""
  byte_vectors = tf.one_hot(features[CHARS_FEATURE], 256, 1., 0.)
  byte_list = tf.unstack(byte_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

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
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  tpu_grpc_url = None
  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    # Determine the gRPC URL of the TPU device to use
    if not FLAGS.master and not FLAGS.tpu_name:
      raise RuntimeError('You must specify either --master or --tpu_name.')

    if FLAGS.master:
      if FLAGS.tpu_name:
        tf.logging.warn('Both --master and --tpu_name are set. Ignoring'
                        ' --tpu_name and using --master.')
      tpu_grpc_url = FLAGS.master
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name,
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
  else:
    # URL is unused if running locally without TPU
    tpu_grpc_url = None
 
  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      cluster=tpu_cluster_resolver,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores))
    
  # Prepare training and testing data
  dbpedia = tf.contrib.learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.DataFrame(dbpedia.test.data)[1]
  y_test = pandas.Series(dbpedia.test.target)

  # Process vocabulary
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(
      MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))

  # Build model
  #classifier = tf.estimator.Estimator(model_fn=char_rnn_model)
  classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=char_rnn_model,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)    


  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={CHARS_FEATURE: x_train},
      y=y_train,
      batch_size=128,
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=100)

  # Eval.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={CHARS_FEATURE: x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False)

  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

  #run CMD: python text_classification_TPU.py --use_tpu=False