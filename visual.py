import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(x, y_, y):
    fig = plt.figure()
    plt.subplot(211)
    im = plt.imshow(x[0], animated=True)
    plt.subplot(212)
    x_vals = np.arange(len(y_))
    plot_y_, = plt.plot([1], [y_[0]])
    plot_y, = plt.plot([1], [y[0]])
    plt.xlim(0, len(y_))
    plt.ylim(0, 1)
    plt.legend(['Real', 'Predicted'])

    def frame(i):
        im.set_array(x[i])
        plot_y_.set_data(x_vals[0:i + 1], y_[0:i + 1])
        plot_y.set_data(x_vals[0:i + 1], y[0:i + 1])
        return im, plot_y_, plot_y

    ani = FuncAnimation(fig, frame, frames=np.arange(1, len(x)), interval=50,
                        repeat=False, blit=True)
    plt.show()


def graph(err, y_, y):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(err)
    plt.legend(['Mean squared error'])
    plt.xlabel('Epoch')
    plt.subplot(212)
    plt.plot(range(len(sample_expected)), sample_expected)
    plt.plot(range(len(sample_expected)), sample_actual)
    plt.legend(['Real', 'Predicted'])
    plt.axis([0, len(sample_expected), 0, 1])
    plt.show()
