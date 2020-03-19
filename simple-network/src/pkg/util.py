import numpy as np
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_output_vector(label, nodes):
    """Return output vector for the given label number"""
    output = np.zeros(nodes) + 0.01
    output[int(label)] = 0.99
    return output
