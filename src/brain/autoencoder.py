from __future__ import print_function
import tensorflow as tf
from brain.deepnet import generateWeightAndBias
from time import time

class Convolutional(object):
    def __init__(self):
        self.parameters = []

    def addConv(self, patch_size, depth, stride):
        self.parameters.append((patch_size, depth, stride))
        

    def modelEncoder(self,input):
        self.input_shape = [int(a) for a in input.get_shape()]
        
        previous_depth = self.input_shape[-1]
        flow = input
        for patch_size,depth,stride in self.parameters:
            w,b = generateWeightAndBias([patch_size,patch_size,previous_depth,depth])
            previous_depth = depth
            flow = tf.nn.conv2d(flow, w, strides=[1, 1, 1, 1], padding='SAME')
            flow = tf.nn.bias_add(flow, b)
            flow = tf.nn.max_pool(flow, ksize=[1,stride,stride,1], strides=[1,stride,stride,1],padding='SAME')
            flow = tf.nn.relu(flow)
        return flow

class AutoEncoder(Convolutional):

    def modelDecoder(self,input):
        flow = input
        batch_size,height,width,depth = [int(a) for a in input.get_shape()]
        parameters = list(reversed(self.parameters))
        for i in range(len(parameters)-1):
            patch_size,depth,stride = parameters[i]
            next_depth = parameters[i+1][1]
            patch_size,depth,stride = parameters[i]
            w = tf.Variable(tf.truncated_normal([patch_size,patch_size,next_depth,depth], stddev=.1))
            b = tf.Variable(tf.constant(.01, shape=[next_depth]))
            height = height*stride
            width = width * stride
            flow = tf.nn.conv2d_transpose(
                flow, w, output_shape=[batch_size,height,width,next_depth], strides=[1, stride, stride, 1], padding="SAME")
            flow = tf.nn.bias_add(flow, b)
            flow = tf.nn.relu(flow)
        patch_size,depth,stride = parameters[-1]
        original_depth = self.input_shape[-1]
        w = tf.Variable(tf.truncated_normal([patch_size,patch_size,original_depth,depth], stddev=.1))
        b = tf.Variable(tf.constant(.01, shape=[original_depth]))
        flow = tf.nn.conv2d_transpose(
            flow, w, output_shape=self.input_shape, strides=[1, stride, stride, 1], padding="SAME")
        flow = tf.nn.bias_add(flow, b)
        return tf.nn.tanh(flow)
                
    def model(self,input):
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
