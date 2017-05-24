""" Failing Convolutional RNN Implementation"""

import tensorflow as tf
import data_source
from model import RadarNet


print('Building training model')
network = RadarNet(20)
x, y, y_, loss, optimize = network.tensors()

inputs, outputs = data_source.load(
    'data', pred_distance=5, pos_x=116, pos_y=116, size=8)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(1000):
        x_seq = inputs[0:20]
        y_seq = outputs[0:20]
        err_total = 0
        session.run(loss, feed_dict={x: x_seq, y_: y_seq})
        for seq in range(1, 15):
            x_seq = inputs[seq * 20:(seq + 1) * 20]
            y_seq = outputs[seq * 20:(seq + 1) * 20]
            err, _ = session.run((loss, optimize),
                                 feed_dict={x: x_seq, y_: y_seq})
            err_total = err_total + err
        print('Epoch {} - avg loss: {}'.format(epoch, err_total / 15))
