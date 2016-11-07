import tensorflow as tf

def generateWeightAndBias(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=.01))
    bias = tf.Variable(tf.constant(.01, shape=[shape[-1]]))
    return weight, bias


class NNLayer(object):
    pass


class NeuralNet(object):
    pass


# class DeepNet(object):
#     def __init__(self):
#         pass

#     def save(self, file):
#         pickle.dump()

#     def load(self, file):
#         pickle.load()
