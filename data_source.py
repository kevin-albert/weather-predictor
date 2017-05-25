""" Reads input / output image data"""

from os import listdir
from os.path import join, exists
import numpy as np


def load_data(data_dir, test_rate=None):
    """
    Read input data from the given directory
    """
    inputs = []
    if exists(join(data_dir, '_cache.npy')):
        inputs = np.load(join(data_dir, '_cache.npy'))
    else:
        inputs = []
        for data_file in sorted(listdir(data_dir)):
            if data_file.endswith('.txt'):
                inputs.append(np.genfromtxt(join(data_dir, data_file)) / 10)
        np.save(join(data_dir, '_cache'), inputs)
    num_train = int(len(inputs) * (1 - test_rate))
    return inputs[0:num_train], inputs[num_train:]


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
        val = 0
        # take min of each 2x2 square and max of that
        for j in range(0, size, 2):
            for k in range(0, size, 2):
                minpool = np.min(pool[j:j + 1, k:k + 1])
                if minpool > val:
                    val = minpool
        outputs = outputs + [val]
    return outputs
