import brain.autoencoder as ae
import numpy as np
from brain.prepros import dataGenerator, showMultipleArraysHorizontally
import tensorflow as tf

batch_size = 8
train_dataset = '../data/processed/notmnist_images_train'
valid_dataset = '../data/processed/notmnist_images_valid'


def _gen(file_name):
    gen = dataGenerator(batch_size, file_name)
    while True:
        yield np.expand_dims(next(gen), axis=3)


t_gen = _gen(train_dataset)
v_gen = _gen(valid_dataset)

# testing the generators
# gen = dataGenerator(80,train_dataset)
# array = next(gen)
# array = next(gen)
# array = next(gen)+.5
# showMultipleArraysHorizontally(array, max_per_row=4)

# net = ae.ConvolutionalAutoencoderSingle()
# net.model()
# net.optimizer()
# net.train(100000, t_gen, 100, v_gen)

with tf.graph().as_default():
    data_input = tf.placeholder(tf.float32, shape=(
                self.batch_size, image_size, image_size, num_channels), name="data_input_placeholder")

    network = ae.Convolutional()
    flow = network.addLayer(data_input, [3, 3, 1, 8], 2,relu=True)
    flow = network.addLayer(flow, [3, 3, 8, 16], 2,relu=True)
    reconstructed = ae.Deconvolutional(network)
    loss = tf.reduce_mean(
                tf.abs(reconstructed - data_input))
    optimizer = tf.train.AdamOptimizer(.001).minimize(loss)