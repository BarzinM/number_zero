# dependenices
from __future__ import print_function
import collections
from perception import Encoder
import numpy as np
from lib.camera import Camera
import socket
import tensorflow as tf
import threading
from time import sleep


buffer_length = 500
temp_buffer_length = 10
batch_size = 1
height = 120
width = 160
body_address = ('192.168.1.4', 8089)
loss_offset = .1
running_ratio = .05
remote_camera = False

# initialize model

buffer = np.zeros((buffer_length, height, width))
data_input = tf.placeholder(tf.float32, shape=(
    batch_size, height, width, 1), name="data_input")
net = Encoder(data_input)
loss = tf.reduce_mean(tf.abs(net.decoded - data_input))
optimizer = tf.train.AdamOptimizer(.01).minimize(loss)


# initialize a numpy buffer
temp_buffer = collections.deque(maxlen=temp_buffer_length)
stream_loss_average = height * width
loss_coef = 1 + loss_offset

cam = Camera()
cam.setSize(width,height)

if remote_camera:
    # connect to body
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(100)
    connection.connect(body_address)

    # receive stream
    cam.receive(connection)
else:
    cam.capture()

session = tf.Session()

def capture():
    frame = np.empty((1,height, width,1), np.float32)
    while True:
        # run through encoder and calculate loss, calculate running average of
        # stream
        frame[0,:, :,0] = cam.getFrame()

        # process received frame
        frame = frame / 255 - .5
        stream_loss = session.run(loss,feed_dict={data_input: frame})
        print(stream_loss)
        sleep(.3)
        # if loss is larger than average of buffer, add to buffer queue
        # if stream_loss > stream_loss_average * loss_coef:
        #     temp_buffer.append(frame)

        # stream_loss_average = stream_loss * running_ratio + \
        #     stream_loss_average * (1 - running_ratio)


with session.as_default():
    tf.initialize_all_variables().run()
    thrd = threading.Thread(target=capture)
    thrd.daemon = True
    thrd.start()
    frame = np.empty((1,height, width,1), np.float32)
    try:
        while True:
            frame[0,:, :,0] = cam.getFrame()
            frame = frame / 255 - .5
            session.run([optimizer],feed_dict={data_input:frame})
    except KeyboardInterrupt:
        pass

cam.close()

#####################



# in another thread, train from thread, calculate loss for each data point
# with tf.Session() as session:
#     tf.initialize_all_variables().run()
#     while True:
#         # get batch from buffer
#         random_indexes = np.randint(buffer_length, batch_size)
#         batch = buffer[random_indexes]
#         feed_dict = {data_input: batch}
#         _ = session.run([optimizer,loss],feed_dict=feed_dict)
#         loss = model.train(batch)

#         # if loss of each is less than running average of stream, substitute with
#         # one from queue
#         if loss < stream_loss_average:
#             data_point = temp_buffer.popleft()
#             buffer[random_indexes] = temp_buffer[:batch_size]
