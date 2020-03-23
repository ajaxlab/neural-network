"""
This demonstrates image data and it's label for sample.csv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pkg.network as net
import pkg.util as util


def main():
    layer1 = 28 * 28
    layer2 = 100
    layer3 = 10
    learning_rate = 0.3

    nn = net.NeuralNet(layer1, layer2, layer3, learning_rate)
    nn.load_model()

    images = []
    for label in range(10):
        target = np.zeros(layer3) + 0.01
        target[label] = 0.99
        image = nn.inverse(target)
        print(image)
        images.append(image)

    fig = plt.figure()
    max_index = len(images) - 1

    def get_pixels(index):
        if index >= max_index:
            plt.close()
        ret = images[index].reshape(28, 28)
        print('index: ', index)
        return ret

    im = plt.imshow(get_pixels(0), cmap='Greys',
                    interpolation='None', animated=True)

    def updatefig(frame, *args):
        im.set_array(get_pixels(frame))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=4000, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
