"""
This demonstrates image data and it's label for sample.csv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pkg.util as util


def main():
    labels = []
    images = []
    with open('dataset/sample.csv') as file:
        for line in file:
            tokens = line.strip().split(',')
            pixels = np.asfarray(tokens[1:])
            labels.append(tokens[0])
            images.append(pixels.reshape(28, 28))

    fig = plt.figure()
    max_index = len(images) - 1

    def get_pixels(index):
        if index >= max_index:
            plt.close()
        ret = images[index]
        label = labels[index]
        print('label: ', label, util.get_output_vector(label, 10))
        return ret

    im = plt.imshow(get_pixels(0), cmap='Greys',
                    interpolation='None', animated=True)

    def updatefig(frame, *args):
        im.set_array(get_pixels(frame))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
