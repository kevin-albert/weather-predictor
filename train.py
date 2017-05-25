""" Failing Convolutional RNN Implementation"""

import tensorflow as tf
import matplotlib.pyplot as plt
import data_source
from model import RadarNet

seq_length = 20
washout = 20
sample = 150
epochs = 100

print('Building model')
network = RadarNet()


print('Loading input data')
inputs = data_source.load('data')
print('Generating target outputs')
outputs = data_source.generate_targets(inputs, 5, (118, 148), 5)
N = len(outputs)

# remove data that we aren't predicting with
inputs = inputs[0:N]
print('Data length: {}'.format(N))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    err_plot = []
    sample_expected = []
    sample_actual = []
    print('Training...')
    for epoch in range(epochs):
        network.reset_state()
        network.washout(session, inputs[0:washout])

        for seq in range(washout, N, seq_length):
            max_length = min(N - seq, seq_length)
            seq_end = seq + max_length
            x_seq = inputs[seq:seq + max_length]
            y_seq = outputs[seq:seq + max_length]
            network.train_seq(session, x_seq, y_seq)

        # sample run
        network.reset_state()
        network.washout(session, inputs[0:washout])
        sample_start = washout
        sample_end = min(N, sample_start + sample)
        x_seq = inputs[sample_start:sample_end]
        y_seq = outputs[sample_start:sample_end]
        y, loss = network.test_seq(session, x_seq, y_seq)

        err_plot.append(loss)
        if epoch % 10 == 0:
            print('[{: >6}] loss: {}'.format(epoch, loss))

        # if this is the last epoch, record the sample run
        if epoch == epochs - 1:
            sample_expected = y_seq
            sample_actual = y

    print('Done')
    plt.figure(1)
    plt.subplot(211)
    plt.plot(err_plot)
    plt.legend('Mean squared error')
    plt.xlabel('Epoch')
    plt.subplot(212)
    plt.plot(range(len(sample_expected)), sample_expected)
    plt.plot(range(len(sample_expected)), sample_actual)
    plt.legend(['Real', 'Predicted'])
    plt.axis([0, len(sample_expected), 0, 1])
    plt.show()
