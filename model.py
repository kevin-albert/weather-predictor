""" CNN / RNN Model """

import tensorflow as tf
import lstm_ops as lstm
import numpy as np


def model(stage='Train', seq_length=20):
    """
    First pass at neural network model
     * Resize image to 64x64x1
     * Convolutional layer:
        * 8x8x1 receptive field
        * 4x4 stride
        * 10 features per block
        * ReLu activation
     *  Convolutional layer:
        * 4x4x10 receptive field
        * 1x1 stride
        * 4 features per block
        * ReLu activation
    * RNN layer
        * 50x1 LSTM cell
        * Fully connected
    * Output layer
        * 1 unit
        * Tanh activation
    """

    # Input manipulation
    x = tf.placeholder(tf.float32, [seq_length, 232 * 232])
    x_img = tf.image.resize_bicubic(
        tf.reshape(x, [-1, 232, 232, 1]),
        [64, 64])

    # CNN Layer
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 10], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[10]))
    h_conv1 = tf.nn.conv2d(x_img, W_conv1, strides=[1, 4, 4, 1],
                           padding='SAME') + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)

    # CNN Layer
    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 10, 4], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[4]))
    h_conv2 = tf.nn.conv2d(h_relu1, W_conv2, strides=[1, 2, 2, 1],
                           padding='SAME') + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)

    # RNN Layer
    h_unstack3 = tf.unstack(tf.reshape(h_relu2, [seq_length, 1, 256]), axis=0)
    h_lstm3 = tf.contrib.rnn.BasicLSTMCell(50)
    h_outputs3, h_state3 = tf.contrib.rnn.static_rnn(
        h_lstm3, h_unstack3, dtype=tf.float32)

    # Output Layer
    W_out4 = tf.Variable(tf.truncated_normal([1, 50], stddev=1 / np.sqrt(50)))
    b_out4 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
    y_unstack = [W_out4 * tf.reshape(h_outputs3[i], [-1]) +
                 b_out4 for i in range(seq_length)]
    y = tf.tanh(tf.stack(y_unstack))

    return x, y
