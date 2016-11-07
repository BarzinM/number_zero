import tensorflow as tf
from brain.deepnet import generateWeightAndBias


class ConvolutionalAutoencoderSingle(object):
    def __init__(self):
        pass

    def model(self):
        self.batch_size = 8
        image_size = 28
        num_channels = 1
        patch_size = 5
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_input = tf.placeholder(tf.float32, shape=(
                self.batch_size, image_size, image_size, num_channels), name="data_input_placeholder")

            weight, bias_1 = generateWeightAndBias(
                [patch_size, patch_size, 1, 5])
            # bias_2 = tf.Variable(
            #     tf.constant(.01, shape=(
                # self.batch_size, image_size, image_size, num_channels)))

            data = tf.nn.conv2d(self.data_input, weight, strides=[
                                1, 1, 1, 1], padding='SAME')

            self.encoded = tf.nn.relu(data + bias_1)

            data = tf.nn.conv2d_transpose(
                self.encoded, weight, output_shape=[self.batch_size, image_size, image_size, num_channels], strides=[1, 1, 1, 1])
            self.reconstructed = tf.nn.tanh(data)

    def optimizer(self):
        with self.graph.as_default():
            self.loss = tf.reduce_mean(
                tf.square(self.reconstructed - self.data_input))
            self.optimizer = tf.train.AdamOptimizer(.001).minimize(self.loss)

    def train(self, train_steps, train_generator, valid_steps, valid_generator):
        saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            for i in range(1, train_steps + 1):
                train_data = next(train_generator)
                _, c = session.run([self.optimizer, self.loss], feed_dict={
                                   self.data_input: train_data})
                if i % 100 == 0:
                    print('train loss is', c,end='')
                    c = 0
                    for batch in range(0, valid_steps):
                        valid_data = next(valid_generator)
                        feed_dict = {self.data_input: valid_data}
                        c += self.loss.eval(feed_dict)
                    print(' Validation loss is', 100 * c /
                          (valid_steps * self.batch_size))
            save_path = saver.save(session,"/tmp/model.ckpt")
            print("Model saved in file: %s"%save_path)

    def loadModel(self,path):
        # https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#exporting-and-importing-meta-graphs
        self.model()
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session,path)


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
