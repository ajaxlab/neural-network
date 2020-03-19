import pkg.network as net
import pkg.util as util
import numpy as np


def main():
    layer1 = 28 * 28
    layer2 = 100
    layer3 = 10
    learning_rate = 0.3

    nn = net.NeuralNet(layer1, layer2, layer3, learning_rate)

    with open('dataset/mnist_train_100.csv') as file:
        for line in file:
            tokens = line.strip().split(',')
            pixels = (np.asfarray(tokens[1:]) / 255.0 * 0.99) + 0.01
            label = util.get_output_vector(tokens[0], layer3)
            nn.train(pixels, label)

    with open('dataset/mnist_test_10.csv') as file:
        for line in file:
            tokens = line.strip().split(',')
            pixels = (np.asfarray(tokens[1:]) / 255.0 * 0.99) + 0.01
            result = nn.query(pixels)
            print(tokens[0], np.argmax(result))


if __name__ == '__main__':
    main()
