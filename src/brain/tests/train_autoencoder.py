import brain.autoencoder as ae
import numpy as np
from brain.prepros import dataGenerator, showMultipleArraysHorizontally
import tensorflow as tf
from time import time

batch_size = 8
train_dataset = '../data/processed/notmnist_images_train'
valid_dataset = '../data/processed/notmnist_images_valid'


def _gen(file_name):
    gen = dataGenerator(batch_size, file_name)
    while True:
        yield np.expand_dims(next(gen), axis=3)


train_generator = _gen(train_dataset)
valid_generator = _gen(valid_dataset)


net = ae.AutoEncoder()
net.addConv(7,4,2)
net.addConv(7,12,7)
# net.addConv(3,12,7)

data_input = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1), name="data_input_placeholder")
output = net.model(data_input)
loss = tf.reduce_mean(tf.abs(output - data_input))
optimizer = tf.train.AdamOptimizer(.001).minimize(loss)


train_steps = 100000
valid_steps = 1000

with tf.Session() as session:
    tf.initialize_all_variables().run()
    start_time = time()
    for i in range(1, train_steps + 1):
        train_data = next(train_generator)
        _, l = session.run([optimizer, loss], feed_dict={
                           data_input: train_data})
        if i % 100 == 0:
            c = 0
            for batch in range(0, valid_steps):
                valid_data = next(valid_generator)
                feed_dict = {data_input: valid_data}
                c += loss.eval(feed_dict)
            print('%5d: Train loss: %.3f, Validation loss: %.3f' %
                  (i, l, c / valid_steps))
    print("Took %f seconds" % (time() - start_time))

