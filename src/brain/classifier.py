from __future__ import print_function
import tensorflow as tf
from brain.deepnet import generateWeightAndBias
from brain.preprocessing import shuffleMultipleArrays, showMultipleArraysHorizontally, skew
import brain.autoencoder as ae
import os
import numpy as np
import glob
from scipy.misc import imread, imrotate
from keras.preprocessing.image import ImageDataGenerator

validation_ratio = .2
batch_size = 16

image_shape = (120, 160)

dir_project = os.environ["NUMBER_ZERO_PATH"]
dir_data = os.path.abspath(os.path.join(dir_project, '../number_zero_dataset'))

positive_files = glob.glob(os.path.join(dir_data, 'positives/*.png'))
negative_files = glob.glob(os.path.join(dir_data, 'negatives/*.png'))

number_of_files = len(positive_files) + len(negative_files)

dataset = np.zeros((number_of_files, *image_shape, 1))

for i, file in enumerate(positive_files + negative_files):
    dataset[i, :, :, 0] = imread(file)

labels = np.concatenate([np.ones((len(positive_files),1)),
                         np.zeros((len(negative_files),1))])

dataset = dataset/255 - .5

dataset,labels = shuffleMultipleArrays([dataset,labels])
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                             rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, shear_range=.2, zoom_range=.2)

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                             rotation_range=20, horizontal_flip=True)


datagen.fit(dataset)


validation_size = 16
train_data = dataset[validation_size:]
train_labels = labels[validation_size:]

print("train:",train_data.shape)

validation_data = dataset[:validation_size]
validation_labels = labels[:validation_size]

train_set = datagen.flow(train_data, train_labels, batch_size=batch_size)

net = ae.Convolutional()
net.addConv(5,2,2)
net.addConv(5,4,2)
net.addConv(5,8,2)
net.addConv(5,16,2)


data_input = tf.placeholder(tf.float32,shape=(batch_size,*image_shape,1),name="data_input")
label_input = tf.placeholder(tf.float32, shape=(batch_size,1),name="label_input")
flow = net.model(data_input)
shape = flow.get_shape().as_list()
print("shape",shape)

flow = tf.reshape(flow,[shape[0],shape[1]*shape[2]*shape[3]])

w,b = generateWeightAndBias((flow.get_shape().as_list()[1],1))

output = tf.nn.sigmoid(tf.matmul(flow,w)+b)

print("output shape",output.get_shape())

error = tf.abs(output-label_input)
print("error shape",error.get_shape())

loss_1 = tf.reduce_mean(error)

loss = loss_1+1e-7*(tf.nn.l2_loss(w)+net.loss)

optimizer = tf.train.AdamOptimizer(10).minimize(loss)

train_steps = 10000
valid_steps = 1

L= 0
with tf.Session() as session:
    tf.initialize_all_variables().run()
    # start_time = time()
    print(session.run(w)[0:5])
            
    for i in range(1, train_steps + 1):
        train_data = next(train_set)
        if len(train_data[1])!=batch_size:
            train_data = next(train_set)
        _, l = session.run([optimizer, loss], feed_dict={
                           data_input: train_data[0],label_input: train_data[1]})
        L += l
        if i % 100 == 0:
            c = 0
            for batch in range(0, valid_steps):
                # valid_data = next(valid_generator)
                feed_dict = {data_input: validation_data[:batch_size],label_input:validation_labels[:batch_size]}
                c += loss_1.eval(feed_dict)
            print(session.run(w)[0:5])
            print('%5d: Train loss: %.3f, Validation loss: %.3f' %
                  (i, L/100, c / valid_steps))
            L = 0
    # print("Took %f seconds" % (time() - start_time))

