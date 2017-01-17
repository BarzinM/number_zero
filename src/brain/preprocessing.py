import numpy as np
from scipy import ndimage
import os
import random
import glob
from scipy.ndimage.interpolation import geometric_transform


def skew(array, shift_size):
    h, l = array.shape
    def mapping(lc):
        l, c = lc
        dec = (dl * (l - h)) / h
        return l, c + dec
    dl = 50
    c = geometric_transform(a, mapping, (h, l + dl), order=5, mode='nearest')


def toOnehot(array, num_classes=10):
    """
    Takes a 1 dimensional array and returns a 2 dimensional array of one-hots with the dimension 0 with the same size of input array and dimension 1 with the size of `num_of_classes`.

    Inputs:
    - array: 1D array in which each element is a class index.
    - num_classes: number of classes that each element of `array` falls into.

    Outputs:
    - A 2D array of one-hots.
    """
    count = len(array)
    onehot_array = np.zeros((count, num_classes), np.int8)
    onehot_array[np.arange(count), array] = 1
    return onehot_array


def showMultipleArraysHorizontally(array, labels=None, max_per_row=10):
    """
    Takes an array with the shape [number of images, width, height] and shows the images in rows.

    Inputs:
    - array: a 3 dimensional array with the shape: [number of images, width, height]
    - labels: a 1 dimensional array with the length of number of images in which each element is the label of corresponding image in input `array`.
    - max_per_row: maximum number of images in each row before going to the next row.
    """
    from matplotlib.pyplot import figure, imshow, axis
    # from matplotlib.image import imread
    # from random import sample
    # from matplotlib.pyplot import matshow
    import matplotlib.pyplot as plt
    fig = figure()
    number_of_images = len(array)
    rows = np.floor(number_of_images / max_per_row) + 1
    columns = min(number_of_images, max_per_row)
    for i in range(number_of_images):
        ax = fig.add_subplot(rows, columns, i + 1)
        if labels is not None:
            ax.set_title(labels[i])
        plt.imshow(array[i], cmap='gray')
        axis('off')
    plt.show()


def dataGenerator(batch_size, file_name):
    file_handle = open(file_name, "rb")
    while True:

        # get data array
        try:
            data = np.load(file_handle)
        # if reached end of file
        except IOError:
            # go to the beginning
            file_handle.seek(0)
            # and try loading again
            data = np.load(file_handle)

        # get batches
        number_of_datapoints = data.shape[0]

        full_batches = number_of_datapoints // batch_size

        for batch_count in range(0, full_batches):
            batch_data = data[batch_count *
                              batch_size:(batch_count + 1) * batch_size]

            yield batch_data


def arrayToFile(file_name, array, batch_size):

    file_handle = open(file_name, "wb")
    number_of_data = array.shape[0]
    print('Saving %d data samples into "%s" ...' % (number_of_data, file_name))

    # iterate over data in big batches
    for batch_start in range(0, number_of_data, batch_size):
        np.save(file_handle, array[batch_start: batch_start + batch_size])

        # process status
        completion_percentil = 100 * \
            min(batch_start + batch_size, number_of_data) / number_of_data
        print("Compeleted %%%d" % completion_percentil)

    # always close the file
    file_handle.close()


def imageToArray(file_name):
    return ndimage.imread(file_name).astype(float)


def fileListToArray(file_list):
    image_data = imageToArray(file_list[0])
    dataset = np.ndarray(shape=(len(file_list), image_data.shape[0], image_data.shape[1]),
                         dtype=np.float32)
    dataset[0, :, :] = image_data

    for i in range(1, len(file_list)):
        try:
            dataset[i, :, :] = imageToArray(file_list[i])
        except IOError:
            pass

    return dataset


def ls(directory, extension='*.*'):
    return glob.glob(os.path.join(directory, extension))


def shuffleFiles(array):
    return random.sample(array, len(array))


def shuffleMultipleArrays(list_of_arrays):
    # assert type(list_of_arrays[0]) == list
    indx = np.random.permutation(len(list_of_arrays[0]))
    return [array[indx] for array in list_of_arrays]


def scaleData(array, depth):
    """
    Scales values of elements in an array that range between [0, depth] into an array with elements between [-0.5, .05]

    Inputs:
    - array: a numpy array with elements between [0, depth]
    - depth: depth of values (e.g. maximum value that any element is allowed to have)

    Output:
    - An array with the same shape of input array with elements between [-0.5, 0.5]
    """
    return array / depth - .5
