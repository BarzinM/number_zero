from __future__ import print_function
import tensorflow as tf
from brain.deepnet import generateWeightAndBias
from time import time


class Encoder(object):
    def __init__(self, input_tensor):
        depth_list = [8, 16, 32, 64]
        stride_list = [2, 2, 2, 2]
        kernel_size_list = [5, 5, 5, 5]

        shape = input_tensor.get_shape().as_list()
        flow_shape = [(shape[1],shape[2])]
        depth_list = [shape[3]]+depth_list
        i=0
        self.input_tensor = input_tensor

        weight_1, bias_1 = generateWeightAndBias(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]])
        self.loss = tf.nn.l2_loss(weight_1)
        flow = tf.nn.conv2d(input_tensor, weight_1, strides=[
                            1, 1, 1, 1], padding='SAME')
        flow = tf.nn.bias_add(flow, bias_1)
        flow = tf.nn.max_pool(flow, ksize=[1, stride_list[i], stride_list[i], 1], strides=[
                              1, stride_list[i], stride_list[i], 1], padding='SAME')
        flow = tf.nn.relu(flow)
        shape = flow.get_shape().as_list()
        flow_shape.append((shape[1],shape[2]))
        
        print(flow.get_shape().as_list())

        i += 1
        weight_2, bias_2 = generateWeightAndBias(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]])
        self.loss += tf.nn.l2_loss(weight_2)
        flow = tf.nn.conv2d(flow, weight_2, strides=[
                            1, 1, 1, 1], padding='SAME')
        flow = tf.nn.bias_add(flow, bias_2)
        flow = tf.nn.max_pool(flow, ksize=[1, stride_list[i], stride_list[i], 1], strides=[
                              1, stride_list[i], stride_list[i], 1], padding='SAME')
        flow = tf.nn.relu(flow)
        shape = flow.get_shape().as_list()
        flow_shape.append((shape[1],shape[2]))

        print(flow.get_shape().as_list())

        i += 1
        weight_3, bias_3 = generateWeightAndBias(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]])
        self.loss += tf.nn.l2_loss(weight_3)
        flow = tf.nn.conv2d(flow, weight_3, strides=[
                            1, 1, 1, 1], padding='SAME')
        flow = tf.nn.bias_add(flow, bias_3)
        flow = tf.nn.max_pool(flow, ksize=[1, stride_list[i], stride_list[i], 1], strides=[
                              1, stride_list[i], stride_list[i], 1], padding='SAME')
        flow = tf.nn.relu(flow)
        shape = flow.get_shape().as_list()
        flow_shape.append((shape[1],shape[2]))

        print(flow.get_shape().as_list())

        i += 1
        weight_4, bias_4 = generateWeightAndBias(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]])
        self.loss += tf.nn.l2_loss(weight_4)
        flow = tf.nn.conv2d(flow, weight_4, strides=[
                            1, 1, 1, 1], padding='SAME')
        flow = tf.nn.bias_add(flow, bias_4)
        flow = tf.nn.max_pool(flow, ksize=[1, stride_list[i], stride_list[i], 1], strides=[
                              1, stride_list[i], stride_list[i], 1], padding='SAME')
        self.encoded = tf.nn.relu(flow)
        shape = flow.get_shape().as_list()
        flow_shape.append((shape[1],shape[2]))

        print(self.encoded.get_shape().as_list())

        shape = self.encoded.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        # flow = tf.reshape(flow,[shape[0],shape[1]*shape[2]*shape[3]])

        weight_t_1 = tf.Variable(tf.truncated_normal(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]], stddev=.1))
        bias_t_1 = tf.Variable(tf.constant(.01, shape=[depth_list[i]]))
        height,width = flow_shape[i]
        flow = tf.nn.conv2d_transpose(
            flow, weight_t_1, output_shape=[batch_size, height, width, depth_list[i]], strides=[1, stride_list[i], stride_list[i], 1], padding="SAME")
        flow = tf.nn.bias_add(flow, bias_t_1)
        flow = tf.nn.relu(flow)

        i -= 1
        weight_t_2 = tf.Variable(tf.truncated_normal(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]], stddev=.1))
        bias_t_2 = tf.Variable(tf.constant(.01, shape=[depth_list[i]]))
        height,width = flow_shape[i]
        
        flow = tf.nn.conv2d_transpose(
            flow, weight_t_2, output_shape=[batch_size, height, width, depth_list[i]], strides=[1, stride_list[i], stride_list[i], 1], padding="SAME")
        flow = tf.nn.bias_add(flow, bias_t_2)
        flow = tf.nn.relu(flow)

        i -= 1
        weight_t_3 = tf.Variable(tf.truncated_normal(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]], stddev=.1))
        bias_t_3 = tf.Variable(tf.constant(.01, shape=[depth_list[i]]))
        height,width = flow_shape[i]
        flow = tf.nn.conv2d_transpose(
            flow, weight_t_3, output_shape=[batch_size, height, width, depth_list[i]], strides=[1, stride_list[i], stride_list[i], 1], padding="SAME")
        flow = tf.nn.bias_add(flow, bias_t_3)
        flow = tf.nn.relu(flow)

        i -= 1
        weight_t_4 = tf.Variable(tf.truncated_normal(
            [kernel_size_list[i], kernel_size_list[i], depth_list[i], depth_list[i+1]], stddev=.1))
        bias_t_4 = tf.Variable(tf.constant(.01, shape=[depth_list[i]]))
        height,width = flow_shape[i]
        flow = tf.nn.conv2d_transpose(
            flow, weight_t_4, output_shape=[batch_size, height, width, depth_list[i]], strides=[1, stride_list[i], stride_list[i], 1], padding="SAME")
        flow = tf.nn.bias_add(flow, bias_t_4)
        self.decoded = tf.nn.tanh(flow)

    def encode(self):
        pass

    def decode(self):
        pass

    def loss(self):
        self.error_loss = tf.reduce_mean(
            tf.abs(self.decoded - self.input_tensor))

    def save(self, session):
        var_1 = tf.Variable(value, name='var_1')
        saver = tf.train.Saver([var_1, var_2])
        saver.save(session, file_name)
        pass

    def load(self, session):
        var_1 = tf.Variable(0, validate_shape=False, name='var_1')
        saver = tf.train.Saver()
        saver.restore(session, file_name)

    def reset():
        tf.reset_default_graph()

    def train(self):
        pass


class Convolutional(object):
    def __init__(self):
        self.parameters = []

    def addConv(self, patch_size, depth, stride):
        self.parameters.append((patch_size, depth, stride))

    def modelEncoder(self, input):
        self.input_shape = [int(a) for a in input.get_shape()]

        previous_depth = self.input_shape[-1]
        flow = input
        self.loss = 0
        for patch_size, depth, stride in self.parameters:
            w, b = generateWeightAndBias(
                [patch_size, patch_size, previous_depth, depth])
            self.loss += tf.nn.l2_loss(w)
            previous_depth = depth
            flow = tf.nn.conv2d(flow, w, strides=[1, 1, 1, 1], padding='SAME')
            flow = tf.nn.bias_add(flow, b)
            flow = tf.nn.max_pool(flow, ksize=[1, stride, stride, 1], strides=[
                                  1, stride, stride, 1], padding='SAME')
            flow = tf.nn.relu(flow)
        return flow


class AutoEncoder(Convolutional):

    def modelDecoder(self, input):
        flow = input
        batch_size, height, width, depth = [int(a) for a in input.get_shape()]
        parameters = list(reversed(self.parameters))
        for i in range(len(parameters) - 1):
            patch_size, depth, stride = parameters[i]
            next_depth = parameters[i + 1][1]
            w = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, next_depth, depth], stddev=.1))
            b = tf.Variable(tf.constant(.01, shape=[next_depth]))
            height = height * stride
            width = width * stride
            flow = tf.nn.conv2d_transpose(
                flow, w, output_shape=[batch_size, height, width, next_depth], strides=[1, stride, stride, 1], padding="SAME")
            flow = tf.nn.bias_add(flow, b)
            flow = tf.nn.relu(flow)
        patch_size, depth, stride = parameters[-1]
        original_depth = self.input_shape[-1]
        w = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, original_depth, depth], stddev=.1))
        b = tf.Variable(tf.constant(.01, shape=[original_depth]))
        flow = tf.nn.conv2d_transpose(
            flow, w, output_shape=self.input_shape, strides=[1, stride, stride, 1], padding="SAME")
        flow = tf.nn.bias_add(flow, b)
        return tf.nn.tanh(flow)

    def model(self, input):
        with tf.variable_scope('encoder'):
            flow = self.modelEncoder(input)
        with tf.variable_scope('decoder'):
            return self.modelDecoder(flow)

    def loadModel(self, path):
        # https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#exporting-and-importing-meta-graphs
        self.model()
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, path)
