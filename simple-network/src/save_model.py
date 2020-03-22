import pkg.network as net
import pkg.util as util
import numpy as np


def main():

    print('Save trained model')

    layer1 = 28 * 28
    layer2 = 100
    layer3 = 10
    learning_rate = 0.1

    nn = net.NeuralNet(layer1, layer2, layer3, learning_rate)
    nn.train('dataset/mnist_train.csv', 4)
    nn.save_model()

    score = []

    util.for_each_record('dataset/mnist_test.csv', (
        lambda label, pixels:
        score.append(label == np.argmax(nn.query(pixels)))
    ))

    score = np.asfarray(score)
    print('\nScore =', score.sum() / score.size)


if __name__ == '__main__':
    main()
