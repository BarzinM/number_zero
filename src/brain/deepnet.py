import tensorflow as tf

def generateWeightAndBias(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=.1))
    bias = tf.Variable(tf.constant(.1, shape=[shape[-1]]))
    return weight, bias


class DeepNet(object):
    def __init__(self):
        pass

    def addComponent(self, model, name):
        with tf.variable_scope(name):
            pass

    def save(self):
        pass

    def load(self):
        pass

    def train(self,loss):
        with tf.Session() as session:
            tf.initialize_all_variables().run()
        pass

