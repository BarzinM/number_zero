import brain.autoencoder as ae
import numpy as np
from brain.prepros import dataGenerator, showMultipleArraysHorizontally

batch_size = 80
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

net = ae.ConvolutionalAutoencoderSingle()
net.model()
net.optimizer()
net.train(10000, t_gen, 100, v_gen)
