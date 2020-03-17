import numpy as np
import scipy.special


class NeuralNet:

    def __init__(
        self, num_1=10, num_2=3, num_3=5,
        learn_rate=0.5, activation=None
    ):
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        self.learn_rate = learn_rate

        self.w_12 = np.random.normal(
            0.0, pow(num_2, -0.5), (num_2, num_1)) - 0.5  # (3, 10)
        self.w_23 = np.random.normal(
            0.0, pow(num_3, -0.5), (num_3, num_2)) - 0.5  # (5, 3)

        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: scipy.special.expit(x)

    def train(self, input, real_data):
        real = np.array(real_data, ndmin=2).T
        out_1, out_2, out_3 = self._get_output(input)
        # dW23 = dW23' - dW23 = a x E3 x O3 (1 - O3) * O2T
        # E3 = REAL_DATA - O3
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

    def _get_output(self, input):
        out_1 = np.array(input, ndmin=2).T  # (10, 1)

        sum_2 = np.dot(self.w_12, out_1)  # (3, 1)
        out_2 = self.activation(sum_2)  # (3, 1)

        sum_3 = np.dot(self.w_23, out_2)  # (5, 1)
        out_3 = self.activation(sum_3)  # (5, 1)

        return out_1, out_2, out_3
