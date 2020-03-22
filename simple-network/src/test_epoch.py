import pkg.network as net
import pkg.util as util
import numpy as np


def main():
    layer1 = 28 * 28
    layer2 = 100
    layer3 = 10
    learning_rate = 0.1

    nn = net.NeuralNet(layer1, layer2, layer3, learning_rate)

    for i in range(1, 21):
        nn.train('dataset/mnist_train_100.csv', i)
        score = []
        util.for_each_record('dataset/mnist_test.csv', (
            lambda label, pixels:
            score.append(label == np.argmax(nn.query(pixels)))
        ))
        score = np.asfarray(score)
        print(i, '\t', score.sum() / score.size)


if __name__ == '__main__':
    main()
