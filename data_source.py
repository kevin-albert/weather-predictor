""" Reads input / output image data"""

from os import listdir
from os.path import join, exists
import numpy as np


def load(data_dir):
    """
    Read input data from the given directory
    """
    if exists('_data_cache.npy'):
        return np.load('_data_cache.npy')
    else:
        inputs = []
        for data_file in listdir(data_dir):
            if data_file.endswith('.txt'):
                inputs.append(np.genfromtxt(join(data_dir, data_file)) / 10)
        np.save('_data_cache', inputs)
    return inputs


def generate_targets(inputs, pred_distance, pos, size):
    """
    Generate target outputs by sampling the max of area at pos

    inputs: format given by load()
    pred_distance: how many sequence steps to look ahead
    pos: array or tuple of (x, y)
    size: int - width & height of square to inspect
    """
    outputs = []
    pos = (pos[0] - size // 2, pos[1] - size // 2)
    for i in range(pred_distance, len(inputs)):
        pool = inputs[i][pos[1]:pos[1] + size, pos[0]:pos[0] + size]
        outputs = outputs + [np.max(pool)]
    return outputs
