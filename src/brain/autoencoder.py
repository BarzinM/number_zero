import tensorflow as tf


class ConvolutionalAutoencoder(object):
    def __init__(self, input_dims):
        self.train_dataset_address = None
        self.previous_channels = input_dims
        self.input_dims = input_dims
        self.encode_parameters = []
        pass

    def addLayer(self, kernel_size, channels, strides=[1, 1, 1, 1], normilization=True):
        shape = [kernel_size, kernel_size, self.previous_channels, channels]
        # use 1/sqrt(dim) for stddev:
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        # https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
        bias = tf.Variable(tf.constant(.05, shape=channels))

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

    def train(self, save_path):
        graph = tf.Graph()
        with graph.as_default():
            input_data = tf.placeholder(
                tf.float32, self.input_dims, name='autoencoder_input')
            cost = self.getCost(input_data)
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for i in range(steps):
                _, c = session.run([optimizer, cost], feed_dict={
                                   input_data: train_data})

            session.run(encoded, feed_dict={input_data: test_data})
