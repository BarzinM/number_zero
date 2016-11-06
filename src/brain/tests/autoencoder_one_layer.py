import src.brain.autoencoder as ae
import src.brain.deepnet as dpn
import numpy as np
# import os


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
        data = dpn.shuffleArrays([data])
        
        # get batches
        number_of_datapoints = data.shape[0]
        full_batches = number_of_datapoints//batch_size # few datapoints are going to waste here

        for batch_start in range(0,full_batches,batch_size):
            batch_data = data[batch_start:batch_start+batch_size]
            
            yield batch_data


batch_size = 8

net = ae.ConvolutionalAutoencoderSingle()
net.model()
net.optimizer()
net.train(1000, t_gen, 100, v_gen)

# net.addLayer(5, 16)

# this_dirtory = os.path.dirname(os.path.realpath('__file__'))
# netfile = os.path.join(this_dirtory, 'one_layer_cnn')

# steps = 100

# net.train(steps, data_gen, save_path=netfile)
