import json
import logging
import numpy as np
import scipy.special
from . import util


class NeuralNet:

    def __init__(self, num_1, num_2, num_3, learn_rate):
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        self.learn_rate = learn_rate
        self.record = 0

        self.w_12 = np.random.normal(
            0.0, pow(num_2, -0.5), (num_2, num_1))  # (100, 748)
        self.w_23 = np.random.normal(
            0.0, pow(num_3, -0.5), (num_3, num_2))  # (10, 100)

        self.activation = lambda x: scipy.special.expit(x)
        self.reverse_activation = lambda x: scipy.special.logit(x)

    def train(self, data_path, epoch=1):
        num_3 = self.num_3
        train_record = self.train_record
        get_output_vector = util.get_output_vector
        for n in range(epoch):
            util.for_each_record(data_path, (
                lambda label, pixels:
                train_record(pixels, get_output_vector(label, num_3))
            ))

    def train_record(self, input, target):
        # self.record += 1
        # logging.info(self.record)
        # logging.info([self.record, "train", len(input), target])

        real = np.array(target, ndmin=2).T
        out_1, out_2, out_3 = self._get_output(input)
        # dW23 = dW23' - dW23 = a x E3 x O3 (1 - O3) * O2T
        # E3 = TARGET - O3
        err_3 = real - out_3
        self.w_23 += self.learn_rate * np.dot(
            (err_3 * out_3 * (1 - out_3)),
            out_2.T
        )
        # dW12 = dW12' - dW12 = a x E2 x O2 (1 - O2) * O1T
        # E2 = W23T * E3
        err_2 = np.dot(self.w_23.T, err_3)
        self.w_12 += self.learn_rate * np.dot(
            (err_2 * out_2 * (1 - out_2)),
            out_1.T
        )

    def query(self, input):
        in_1, out_2, out_3 = self._get_output(input)
        return out_3

    def save_model(self):
        np.save('model/w_12.npy', self.w_12)
        np.save('model/w_23.npy', self.w_23)

    def load_model(self):
        self.w_12 = np.load('model/w_12.npy')
        self.w_23 = np.load('model/w_23.npy')

    def inverse(self, output):
        out_3 = np.array(output, ndmin=2).T
        sum_3 = self.reverse_activation(out_3)
        out_2 = self._normalize(np.dot(self.w_23.T, sum_3))
        sum_2 = self.reverse_activation(out_2)
        out_1 = self._normalize(np.dot(self.w_12.T, sum_2))
        return out_1

    def _get_output(self, input):
        out_1 = np.array(input, ndmin=2).T  # (748, 1)

        sum_2 = np.dot(self.w_12, out_1)  # (100, 1)
        out_2 = self.activation(sum_2)  # (100, 1)

        sum_3 = np.dot(self.w_23, out_2)  # (10, 1)
        out_3 = self.activation(sum_3)  # (10, 1)

        return out_1, out_2, out_3

    def _normalize(self, matrix):
        # Normalizes to 0.01 ~ 0.99
        matrix -= np.min(matrix)
        matrix /= np.max(matrix)
        matrix *= 0.98
        matrix += 0.01
        return matrix
