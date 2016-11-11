from __future__ import print_function
import tensorflow as tf
from brain.deepnet import generateWeightAndBias
from time import time


def conv(x, W, b, k=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def deconv(prev_layer, w, b, output_shape, k=2):
    # Deconv layer
    deconv = tf.nn.conv2d_transpose(
        prev_layer, w, output_shape=output_shape, strides=[1, k, k, 1], padding="SAME")
    deconv = tf.nn.bias_add(deconv, b)

    return deconv


class Convolutional(object):
    def __init__(self):
        self.parameters = []
        pass

    def addLayer(self, input, shape, stride,relu=True):
        w, b = generateWeightAndBias(shape)
        self.parameters.append((w, b, stride, input.get_shape()))
        
        x = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.max_pool(x, ksize=strides, strides=strides,
                              padding='SAME')
        if relu:
            x = tf.nn.relu(x)
        self.output = x
        return x

def Deconvolutional(object):
    def __init__(self, conv=None):
        self.parameters = []
        if conv is not None:
            output = conv.output
            for w, b, stride, shape  in reverse(conv.parameters):
                shape[2],shpae[3] = shape[3],shape[2]
                output = self.addLayer(output,w.get_shape(),stide, shape)
            return output

    def addLayer(self, input, shape, strides, output_shape, relu=True):
        b = tf.Variable(tf.constant(.01, shape=[shape[3]]))
        shape[2],shape[3] = shape[3],shape[2]
        w = tf.Variable(tf.truncated_normal(shape, stddev=.1))

        x = tf.nn.conv2d_transpose(
            input, w, output_shape=output_shape, strides=strides, padding="SAME")
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
        self.output = x
        return x

class ConvolutionalAutoencoderSingle(object):
    def __init__(self):
        pass

    def encoder(self, input,):
        with tf.name_scope("encoder") as scope:

    def model(self, depth, patch):
        self.batch_size = 8
        image_size = 28
        num_channels = 1
        patch_size = 5
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_input = tf.placeholder(tf.float32, shape=(
                self.batch_size, image_size, image_size, num_channels), name="data_input_placeholder")

            weights = {
                "conv1": tf.Variable(tf.truncated_normal([3, 3, 1, 8], stddev=.1)),
                "conv2": tf.Variable(tf.truncated_normal([3, 3, 8, 16], stddev=.1)),
                # "conv3": tf.Variable(tf.truncated_normal([3, 3, 16, 16],stddev=.1)),
                "deconv2": tf.Variable(tf.truncated_normal([3, 3, 8, 16], stddev=.1)),
                "deconv1": tf.Variable(tf.truncated_normal([3, 3, 1, 8], stddev=.1))}

            biases = {
                "conv1": tf.Variable(tf.constant(.01, shape=[8])),
                "conv2": tf.Variable(tf.constant(.01, shape=[16])),
                "conv3": tf.Variable(tf.constant(.01, shape=[16])),
                "deconv2": tf.Variable(tf.constant(.01, shape=[8])),
                "deconv1": tf.Variable(tf.constant(.01, shape=[1]))}

            # w,b = generateWeightAndBias([32*7*7,1*28*28])

            data = conv(self.data_input, weights['conv1'], biases['conv1'])
            self.encoded = conv(data, weights['conv2'], biases['conv2'])
            # data = tf.reshape(self.encoded,[8,32*7*7])

            # data = tf.nn.tanh(tf.matmul(data, w)+b)
            # self.reconstructed = tf.reshape(data,[8,28,28,1])
            # self.encoded = conv(data, weights['conv3'], biases['conv3'])

            data = deconv(self.encoded, weights['deconv2'], biases['deconv2'], [
                          self.batch_size, image_size // 2, image_size // 2, 16])
            data = tf.nn.relu(data)
            data = deconv(data, weights['deconv1'], biases['deconv1'], [
                          self.batch_size, image_size, image_size, 1])
            self.reconstructed = tf.nn.tanh(data)

            # http://stackoverflow.com/questions/36548736/tensorflow-unpooling
            # def max_pool(inp, k=2):
            # return tf.nn.max_pool_with_argmax_and_mask(inp, ksize=[1, k, k,
            # 1], strides=[1, k, k, 1], padding="SAME")

            # conv1 = conv2d(x, "conv1")
            # maxp1, maxp1_argmax, maxp1_argmax_mask  = max_pool(conv1)

            # conv2 = conv2d(maxp1, "conv2")
            # maxp2, maxp2_argmax, maxp2_argmax_mask  = max_pool(conv2)

            # conv3 = conv2d(maxp2, "conv3")

            # maxup2 = max_unpool(conv3, maxp2_argmax, maxp2_argmax_mask)
            # deconv2= conv2d_transpose(maxup2, "deconv2", p)

            # maxup1 = max_unpool(deconv2, maxp1_argmax, maxp1_argmax_mask)
            # deconv1 = conv2d_transpose(maxup1, "deconv1", None)

            # weight_1, bias_1 = generateWeightAndBias(
            #     [patch_size, patch_size, 1, 5])

            # weight_2, bias_2 = generateWeightAndBias(
            #     [patch_size, patch_size, 5, 16])

            # bias_3 = tf.Variable(
            #     tf.constant(.01, shape=(
            #         self.batch_size, image_size // 2, image_size // 2, 5)))

            # bias_4 = tf.Variable(
            #     tf.constant(.01, shape=(
            #         self.batch_size, image_size, image_size, num_channels)))
            ##############
            # data = tf.nn.conv2d(self.data_input, weight_1, strides=[
            #                     1, 1, 1, 1], padding='SAME')
            # data = tf.nn.max_pool(data, [1, 2, 2, 1], [
            #     1, 2, 2, 1], padding='SAME')

            # data = tf.nn.relu(data + bias_1)

            ##################
            # data = tf.nn.conv2d(data, weight_2, strides=[
            #                     1, 1, 1, 1], padding='SAME')
            # data = tf.nn.max_pool(data, [1, 2, 2, 1], [
            #     1, 2, 2, 1], padding='SAME')

            # self.encoded = tf.nn.relu(data + bias_2)
            # ####################
            # data = tf.nn.conv2d_transpose(
            #     self.encoded, weight_2, output_shape=[self.batch_size, image_size // 2, image_size // 2, 5], strides=[1, 2, 2, 1])
            # data = tf.nn.relu(data+bias_3)
            # ####################
            # data = tf.nn.conv2d_transpose(
            #     data, weight_1, output_shape=[self.batch_size, image_size, image_size, num_channels], strides=[1, 2, 2, 1])
            # data = data+bias_4
            # ####################
            # self.reconstructed = tf.nn.sigmoid(data)

    def optimizer(self):
        with self.graph.as_default():
            self.loss = tf.reduce_mean(
                tf.abs(self.reconstructed - self.data_input))
            self.optimizer = tf.train.AdamOptimizer(.001).minimize(self.loss)

    def train(self, train_steps, train_generator, valid_steps, valid_generator):
        # saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            start_time = time()
            for i in range(1, train_steps + 1):
                train_data = next(train_generator)
                _, l = session.run([self.optimizer, self.loss], feed_dict={
                                   self.data_input: train_data})
                if i % 100 == 0:
                    c = 0
                    for batch in range(0, valid_steps):
                        valid_data = next(valid_generator)
                        feed_dict = {self.data_input: valid_data}
                        c += self.loss.eval(feed_dict)
                    print('%5d: Train loss: %.3f, Validation loss: %.3f' %
                          (i, l, c / valid_steps))
            print("Took %f seconds" % (time() - start_time))
            # save_path = saver.save(session,"/tmp/model.ckpt")
            # print("Model saved in file: %s"%save_path)

    def loadModel(self, path):
        # https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#exporting-and-importing-meta-graphs
        self.model()
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, path)


class ConvolutionalAutoencoder(object):
    def __init__(self, input_dims, save_path):
        self.train_dataset_address = None
        self.previous_channels = input_dims
        self.input_dims = input_dims
        self.encode_parameters = []
        self.learning_rate = .5
        self.save_path = None
        pass

    def addLayer(self, kernel_size, channels, strides=[1, 1, 1, 1], normilization=True):
        shape = [kernel_size, kernel_size, self.previous_channels, channels]
        # use 1/sqrt(dim) for stddev:
        weight, bias = generateWeightAndBias(shape)
        # https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py

        self.encode_parameters.append([weight, bias, strides, shape])

        self.previous_channels = channels

    def encode(self, input):
        data = input
        for weight, bias, strides, shape in self.encode_parameters:
            data = tf.nn.conv2d(data, weight, strides, padding='SAME')
            data = tf.nn.relu(data + bias)

        self.encoded = data

    def decode(self, input):
        data = input
        for weight, bias, strides, shape in reversed(self.encode_parameters):
            output_shape = [d * s for d, s in zip(data.get_shape(), strides)]
            data = tf.nn.conv2d_transpose(
                data, weight, output_shape=output_shape, strides=strides)
            bias = tf.Variable(tf.constant(.05, shape=output_shape[-1]))
            data = tf.nn.relu(data + bias)
            # for upsampling:
            # http://stackoverflow.com/questions/36728239/bilinear-upsample-in-tensorflow

    def getCost(self, input):
        data = self.encode(input)
        data = self.decode(data)

        return tf.reduce_mean(tf.square(data - input))

    def train(self, steps, learning_rate=None, save_path=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if save_path is not None:
            self.save_path = save_path
        graph = tf.Graph()
        with graph.as_default():
            input_data = tf.placeholder(
                tf.float32, self.input_dims, name='autoencoder_input')
            cost = self.getCost(input_data)
            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for i in range(steps):
                _, c = session.run([optimizer, cost], feed_dict={
                                   input_data: train_data})

            session.run(self.encoded, feed_dict={input_data: test_data})
