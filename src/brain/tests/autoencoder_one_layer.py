import src.brain.autoencoder as ae
import os

batch_size = 8

net = ae.ConvolutionalAutoencoder([batch_size, 28, 28, 1])

net.addLayer(5, 16)

this_dirtory = os.path.dirname(os.path.realpath('__file__'))
netfile = os.path.join(this_dirtory, 'one_layer_cnn')

net.train(steps, data_gen, save_path=netfile)
