from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

f = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """接受的参数为 MNIST 特征数据、标签和模式（来自 tf.estimator.ModeKeys：TRAIN、EVAL、PREDICT）；
  配置 CNN，然后返回预测、损失和训练操作

  配置的模型为：
  Convolutional Layer 1：应用 32 个 5x5 过滤器（提取 5x5 像素的子区域），并应用 ReLU 激活函数
  Pooling Layer 1：使用 2x2 过滤器和步长 2（指定不重叠的池化区域）执行最大池化运算
  Convolutional Layer 2：应用 64 个 5x5 过滤器，并应用 ReLU 激活函数
  Pooling Layer 2：同样，使用 2x2 过滤器和步长 2 执行最大池化运算
  Dense Layer 1：包含 1024 个神经元，其中丢弃正则化率为 0.4（任何指定元素在训练期间被丢弃的概率为 0.4）
  Dense Layer 2（对数层）：包含 10 个神经元，每个数字目标类别 (0–9) 对应一个神经元
  """
  # Input Layer,input_layer 的形状为 [batch_size, 28, 28, 1]
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1, conv1的形状为 [batch_size, 28, 28, 32]
  #padding 参数指定以下两个枚举值之一（不区分大小写）：valid（默认值）或 same
  #该参数指定输出张量与输入张量是否具有相同的高度和宽度值，即设置filter的步长strides
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1,pool1的形状为 [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2,conv2 的形状为 [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2,pool2 的形状为 [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  #reshape,pool2_flat 的形状为 [batch_size, 3136]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer,dense 的形状为 [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  #dropout 的形状为 [batch_size, 1024]
  #rate 参数指定丢弃率,该值表示 40% 的元素会在训练期间被随机丢弃
  #training 参数采用布尔值，表示模型目前是否在训练模式下运行；只有在 training 为 True 的情况下才会执行丢弃操作。
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer,logits 的形状为 [batch_size, 10]
  #该对数层的默认激活函数为线性激活函数
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      #tf.argmax 函数查找该元素的索引,axis 参数指定要沿着 input 张量的哪个轴查找最大值
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      #tf.nn.softmax 应用 softmax 激活函数，以从对数层中得出概率
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  # 在 Estimator 中，我们输入必须是一个函数cnn_model_fn，这个函数必须返回特征和标签（或者只有特征）
  """
    该函数需要返回一个定义好的 tf.estimator.EstimatorSpec 对象，对于不同的 mode，所必须提供的参数是不一样的：
    训练模式，即 mode == tf.estimator.ModeKeys.TRAIN，必须提供的是 loss 和 train_op。
    验证模式，即 mode == tf.estimator.ModeKeys.EVAL，必须提供的是 loss。
    预测模式，即 mode == tf.estimator.ModeKeys.PREDICT，必须提供的是 predicitions。
  """
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  # 将训练特征数据和标签分别传递到 x（作为字典）和 y。
  # batch_size 设置为 100（这意味着模型会在每一步训练 100 个小批次样本）
  # num_epochs=None 表示模型会一直训练，直到达到指定的训练步数。见下面train方法的调用时设置的steps参数
  # shuffle=True，表示随机化处理训练数据
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  # steps=20000（这意味着模型总共要训练 20000 步）
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
