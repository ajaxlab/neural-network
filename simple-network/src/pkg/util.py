import numpy as np
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_output_vector(label, nodes):
    """Return output vector for the given label number"""
    output = np.zeros(nodes) + 0.01
    output[int(label)] = 0.99
    return output


def for_each_record(path, callback):
    """Open path and read each lines
    Then call the callback function with label and pixels.
    """
    with open(path) as file:
        for line in file:
            tokens = line.strip().split(',')
            label = int(tokens[0])
            pixels = (np.asfarray(tokens[1:]) / 255.0 * 0.99) + 0.01
            callback(label, pixels)
