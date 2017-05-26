""" show stuff """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_source import load_data

plt.xkcd()

def animate(inputs):
    # hack together an empty frame of data and set the min/max on the image
    empty_data = np.zeros((242, 242))
    empty_data = np.ma.masked_where(True, empty_data)

    some_data = np.ma.masked_outside(inputs, -75, 75)
    vmin = some_data.min()
    vmax = some_data.max()
    # empty_data = np.ma.masked_outside(empty_data, -75, 75)

    print(vmin,vmax)

    # create the plot
    fig = plt.figure()
    im = plt.imshow(empty_data, vmin=vmin, vmax=vmax, origin='lower', animated=True)
    fig.colorbar(im)

    def init():
        im.set_array(empty_data)
        return im,

    def frame(data):
        im.set_array(np.ma.masked_outside(data, -75, 75))
        return im,

    ani = FuncAnimation(fig, frame, init_func=init, frames=inputs, interval=50,
                        repeat=True, blit=True)
    plt.show()

print('Loading input data')
train_inputs, test_inputs = load_data('data', test_rate=0)

# print(np.zeros((241, 241)))

animate(train_inputs)
