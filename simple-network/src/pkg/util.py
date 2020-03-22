import imageio
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_normalized_pixels(arr):
    """Return gray color value between 0.01 ~ 1.0"""
    return (arr / 255.0 * 0.99) + 0.01


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
            pixels = get_normalized_pixels(np.asfarray(tokens[1:]))
            callback(label, pixels)


def for_each_image_in_path(path, callback):
    """For each image in the given path
    call the callback function with label and pixels.
    """
    for i in range(10):
        img_array = imageio.imread(path + str(i) + '.png', as_gray=True)
        img_data = 255 - img_array.reshape(784)
        pixels = get_normalized_pixels(img_data)
        callback(i, pixels)
