""" CNN / RNN Model """

import tensorflow as tf
import numpy as np


class RadarNet:

    def __init__(self):
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
        x = tf.placeholder(tf.float32, shape=[None, 232, 232])
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
        h_reshape3 = tf.reshape(h_relu2, [-1, 1, 256])
        h_cellstate3 = tf.placeholder(tf.float32, [1, 50])
        h_hiddenstate3 = tf.placeholder(tf.float32, [1, 50])
        h_state3 = tf.contrib.rnn.LSTMStateTuple(
            h_cellstate3, h_hiddenstate3)
        h_lstm3 = tf.contrib.rnn.BasicLSTMCell(50)
        h_outputs3, h_state3 = tf.nn.dynamic_rnn(
            h_lstm3, h_reshape3, dtype=tf.float32, initial_state=h_state3, time_major=True)

        # Output Layer
        W_out4 = tf.Variable(tf.truncated_normal(
            [50, 1], stddev=1 / np.sqrt(50)))
        b_out4 = tf.Variable(tf.truncated_normal([1], stddev=0.1))

        y = tf.tanh(tf.reshape(h_outputs3, [-1, 50]) @ W_out4 + b_out4)

        # Loss, optimization
        y_ = tf.placeholder(tf.float32, shape=[None])
        loss = tf.reduce_mean(tf.squared_difference(
            y, tf.reshape(y_, [-1, 1])))
        optimize = tf.train.AdamOptimizer().minimize(loss)

        self.x = x
        self.c = h_cellstate3
        self.h = h_hiddenstate3
        self.y = y
        self.y_ = y_
        self.loss = loss
        self.optimize = optimize
        self.c_values = np.zeros([1, 50])
        self.h_values = np.zeros([1, 50])

    def washout(self, session, x):
        data = {self.x: x, self.c: self.c_values, self.h: self.h_values}
        targets = (self.c, self.h, self.y)
        c, h, y = session.run(targets, feed_dict=data)
        self.c_values = c
        self.h_values = h

    def train_seq(self, session, x, y_):
        """
        Train against input / output sequences
        session: tensorflow session
        x: input sequence
        y_: output targets

        returns outputs and average loss
        """
        data = {self.x: x, self.y_: y_,
                self.c: self.c_values, self.h: self.h_values}
        targets = (self.optimize, self.loss, self.c, self.h, self.y)
        _, loss, c, h, y = session.run(targets, feed_dict=data)
        self.c_values = c
        self.h_values = h
        return y, loss

    def test_seq(self, session, x, y_):
        """
        Test with a given input / output sequences
        session: tensorflow session
        x: input sequence
        y_: output targets

        returns average loss for sequence
        """
        data = {self.x: x, self.y_: y_,
                self.c: self.c_values, self.h: self.h_values}
        targets = (self.loss, self.c, self.h, self.y)
        loss, c, h, y = session.run(targets, feed_dict=data)
        self.c_values = c
        self.h_values = h
        return y, loss

    def reset_state(self):
        self.c_values = np.zeros([1, 50])
        self.h_values = np.zeros([1, 50])
