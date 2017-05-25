""" Failing Convolutional RNN Implementation"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import visual
from data_source import load_data, generate_targets
from model import RadarNet

# predict this many frames in advance
pred_distance = 5

# predict for this area
target_size = 5
target_location = (118, 148)

# sequence length for bptt
seq_length = 20

# iterate through this many frames before training
washout = 10

# fraction of frames to reserve for testing
test = 0.1

# iterate over training data this many times
epochs = 100

print('Building model')
network = RadarNet()

print('Loading input data')
train_inputs, test_inputs = load_data('data', test_rate=test)
print('Generating target outputs')
train_outputs = generate_targets(train_inputs, pred_distance,
                                 target_location, target_size)
test_outputs = generate_targets(test_inputs, pred_distance,
                                target_location, target_size)

train_inputs = train_inputs[0:len(train_outputs)]
test_inputs = test_inputs[0:len(test_outputs)]


print('Data length: train={}, test={}'.format(
    len(train_outputs), len(test_outputs)))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    err_plot = []
    sample_inputs = []
    sample_expected = []
    sample_actual = []
    print('Training...')
    for epoch in range(epochs):
        network.reset_state()
        network.washout(session, train_inputs[0:washout])

        # train
        for seq in range(washout, len(train_outputs), seq_length):
            max_length = min(len(train_outputs) - seq, seq_length)
            seq_end = seq + max_length
            x_seq = train_inputs[seq:seq + max_length]
            y_seq = train_outputs[seq:seq + max_length]
            network.train_seq(session, x_seq, y_seq)

        # sample
        network.reset_state()
        network.washout(session, test_inputs[0:washout])
        x_seq = test_inputs[washout:]
        y_seq = test_outputs[washout:]
        y, loss = network.test_seq(session, x_seq, y_seq)
        err_plot.append(loss)
        if epoch % 10 == 0:
            print('[{: >6}] loss: {}'.format(epoch, loss))

        # if this is the last epoch, record the sample run
        if epoch == epochs - 1:
            sample_expected = y_seq
            sample_actual = y
            sample_inputs = x_seq

    print('Done')
    visual.animate(x_seq, y_seq, y)
