from __future__ import print_function
import tensorflow as tf
from brain.deepnet import generateWeightAndBias
from time import time


class Encoder(object):
    def __init__(self, shape=(120, 160, 1)):
        self.depth_list = [16, 64, 128, 256]
        self.core_dimensions = []
        kernel_size_list = [5, 5, 5, 5]
        self.stride_list = [2, 2, 2, 2]

        assert len(shape) == 3
        height, width, depth = shape
        self.shape_list = [[height, width]]
        self.parameters = []
        self.depth_list = [depth] + self.depth_list
        for i in range(len(self.depth_list) - 1):
            weight_1, bias_1 = generateWeightAndBias(
                [kernel_size_list[i], kernel_size_list[i], self.depth_list[i], self.depth_list[i + 1]])
            self.parameters.append([weight_1, bias_1])
            height = height // self.stride_list[i]
            width = width // self.stride_list[i]
            self.shape_list.append([height, width])

        self.core_dimensions = [height * width *
                                self.depth_list[-1]] + self.core_dimensions
        # for i in range(len(self.core_dimensions) - 1):
        #     weight_core_1 = tf.Variable(tf.truncated_normal(
        #         [self.core_dimensions[i], self.core_dimensions[i + 1]], stddev=.05))
        #     bias_core_1 = tf.Variable(
        #         tf.constant(.05, shape=[self.core_dimensions[i + 1]]))
        #     self.parameters.append([weight_core_1, bias_core_1])

        # for i in range(len(self.core_dimensions) - 1):
        #     j = len(self.core_dimensions) - i - 1
        #     wieght_core_t_1 = tf.Variable(tf.truncated_normal(
        #         [self.core_dimensions[-i - 1], self.core_dimensions[-i - 2]], stddev=.05))
        #     bias_core_t_1 = tf.Variable(
        #         tf.constant(.05, shape=[self.core_dimensions[-1 - 2]]))
        #     self.parameters.append([wieght_core_t_1, bias_core_t_1])

        for i in range(len(self.depth_list) - 1):
            weight_t_1 = tf.Variable(tf.truncated_normal(
                [kernel_size_list[-i - 1], kernel_size_list[-i - 1], self.depth_list[-i - 2], self.depth_list[-i - 1]], stddev=.05))
            bias_t_1 = tf.Variable(
                tf.constant(.05, shape=[self.depth_list[-i - 2]]))
            self.parameters.append([weight_t_1, bias_t_1])

        # self.l2_loss = tf.nn.l2_loss(weight_1)

    def model(self, input_tensor):

        flow = input_tensor
        shape = input_tensor.get_shape().as_list()
        flow_shape = [(shape[1], shape[2])]
        batch_size = tf.shape(input_tensor)[0]

        for i in range(len(self.depth_list) - 1):
            weight_1, bias_1 = self.parameters[i]
            flow = tf.nn.conv2d(flow, weight_1, strides=[
                                1, 1, 1, 1], padding='SAME')
            flow = tf.nn.bias_add(flow, bias_1)
            flow = tf.nn.max_pool(flow, ksize=[1, self.stride_list[i], self.stride_list[i], 1], strides=[
                                  1, self.stride_list[i], self.stride_list[i], 1], padding='SAME')
            flow = tf.nn.relu(flow)
            shape = flow.get_shape().as_list()
            flow_shape.append((shape[1], shape[2]))

        self.conv = flow
        print("Dimensions of convolutional tensor:",
              self.conv.get_shape().as_list())
        if len(self.core_dimensions) > 1:
            # flow = tf.reshape(flow, [batch_size, flow_shape[-1][0]*flow_shape[-1][0] * self.depth_list[-1]])
            # for i in range()
            # flow = tf.nn.relu(tf.matmul(flow, weight_core_1) + bias_core_1)

            # flow = tf.nn.relu(tf.matmul(flow, wieght_core_t_1) + bias_core_t_1)

            # flow = tf.reshape(flow, [batch_size, shape[1], shape[2], shape[3]])
            pass

        else:
            pass

        for i in range(len(self.depth_list) - 2):
            j = i + len(self.depth_list) - 1 + 2 * \
                (len(self.core_dimensions) - 1)
            w, b = self.parameters[j]
            print("ddddddd:",w.get_shape().as_list(),flow_shape[-i - 1][0], flow_shape[-i - 1][1])
            flow = tf.nn.conv2d_transpose(
                flow, w, output_shape=[batch_size, flow_shape[-i - 2][0], flow_shape[-i - 2][1], self.depth_list[-i - 2]], strides=[1, self.stride_list[-i - 1], self.stride_list[-i - 1], 1], padding="SAME")
            print(flow.get_shape().as_list())
            flow = tf.nn.bias_add(flow, b)
            flow = tf.nn.relu(flow)

        w, b = self.parameters[-1]
        height, width = self.shape_list[0]
        flow = tf.nn.conv2d_transpose(
            flow, w, output_shape=[batch_size, height, width, self.depth_list[0]], strides=[1, self.stride_list[0], self.stride_list[0], 1], padding="SAME")
        flow = tf.nn.bias_add(flow, b)
        self.decoded = tf.nn.sigmoid(flow)

    def encode(self):
        pass

    def decode(self):
        pass

    def loss(self):
        self.error_loss = tf.reduce_mean(
            tf.abs(self.decoded - self.input_tensor))

    def save(self, session):
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
