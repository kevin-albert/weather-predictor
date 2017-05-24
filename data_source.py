""" Reads input / output image data"""

from os import listdir
from os.path import join
import numpy as np


def load(data_dir, pred_distance=15, pos_x=116, pos_y=116, size=8):
    """
    Read input data from the given location
    Generate output data for each frame at t+pred_distance
    """
    print('Loading data from {}...'.format(data_dir))
    inputs = []
    outputs = []
    pos_x = pos_x - int(size / 2)
    pos_y = pos_y - int(size / 2)
    for data_file in listdir(data_dir):
        inputs = inputs + [np.genfromtxt(join(data_dir, data_file)) / 8]
    for i in range(pred_distance, len(inputs)):
        pool = inputs[i][pos_y:pos_y + size, pos_x:pos_x + size]
        outputs = outputs + [np.max(pool)]
    print('Done')
    return inputs, outputs
