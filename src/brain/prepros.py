import numpy as np
from scipy import ndimage
import os


def dataGenerator(batch_size,file_name):
    file_handle = open(file_name, "rb")
    while True:

        # get data array
        try:
            data = np.load(file_handle)
        # if reached end of file
        except OSError:
            # go to the beginning
            file_handle.seek(0)
            # and try loading again
            data = np.load(file_handle)
        
        # randomize
        data = shuffleArrays([data])
        
        # get batches
        number_of_datapoints = data.shape[0]
        full_batches = number_of_datapoints//batch_size # few datapoints are going to waste here

        for batch_start in range(0,full_batches,batch_size):
            batch_data = data[batch_start:batch_start+batch_size]
            
            yield batch_data


def arrayToFile(array,file_name):
    # read lots of files
    pickle_file = dataset+"_preprocessed"
    struct_file = dataset+"/digitStruct.mat"
    number_of_files = sv.getNumberOfFiles(struct_file)
#     number_of_files = big_batch_size # just for debug
    data_samples = np.random.permutation(number_of_files)
    file_handle = open(pickle_file,"wb")

    # iterate over data in big batches
    for batch_start in range(0,number_of_files, big_batch_size):

        # read the .mat file and parse attributes of data files
        batch_indexes = data_samples[batch_start:batch_start+big_batch_size]

        file_names,train_labels = sv.getLabels(struct_file,batch_indexes)
        train_values = sv.getImage(file_names, dataset,shape=image_shape)


        # form and normalize
        pixel_depth = 255
        train_values = sv.scaleData(train_values,pixel_depth)
        train_labels = sv.parseLabels(train_labels,max_digits_in_label)

        # save in file
        np.save(file_handle, train_values)
        np.save(file_handle, train_labels)

        # process status
        completion_percentil = 100*(batch_start+big_batch_size)/number_of_files
        print("Compeleted %%%d"%completion_percentil)

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
        dataset[i, :, :] = imageToArray(file_list[i])

    return dataset


def ls(directory):
    return os.listdir(directory)


def shuffleArrays(list_of_arrays):
    assert type(list_of_arrays[0]) == list
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
