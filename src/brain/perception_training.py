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
filled = 0
temp_buffer_length = 10
batch_size = None
height = 120
width = 160
body_address = ('192.168.1.4', 8089)

FLAG = {'avg': 1.}


def runningMean(new,ratio=.1):
    FLAG['avg'] = ratio * new + (1. - ratio) * FLAG['avg']
    

loss_offset = .1
running_ratio = .05
remote_camera = False


# initialize model
buffer = np.zeros((buffer_length, height, width,1))
data_input = tf.placeholder(tf.float32, shape=(
    batch_size, height, width, 1), name="data_input")
net = Encoder(data_input)
loss = tf.reduce_mean(tf.abs(net.decoded - data_input))
optimizer = tf.train.AdamOptimizer(.001).minimize(loss)


# initialize a numpy buffer
temp_buffer = collections.deque(maxlen=temp_buffer_length)
stream_loss_average = height * width
loss_coef = 1 + loss_offset

cam = Camera()
cam.setSize(width, height)

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

while filled < buffer_length:
    buffer[filled, :, :, 0] = cam.getFrame() / 255 - .5
    filled += 1
print("Buffer filled with initial.")
session = tf.Session()


def capture():
    frame = np.empty((1, height, width, 1), np.float32)
    while True:
        # run through encoder and calculate loss, calculate running average of
        # stream
        frame[0, :, :, 0] = cam.getFrame()

        # process received frame
        frame = frame / 255 - .5
        stream_loss = session.run(loss, feed_dict={data_input: frame})
        if stream_loss > FLAG['avg']:
            # print("adding one to buffer:",stream_loss)
            temp_buffer.append(frame)
        # print(stream_loss)
        sleep(3)
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
    frame = np.empty((1, height, width, 1), np.float32)
    try:
        while True:
            for _ in range(100):
                indexes = np.random.randint(buffer_length, size=16)
                batch = buffer[indexes, :, :, :]
                _, l = session.run([optimizer, loss], feed_dict={
                    data_input: batch})
                runningMean(l)
                frame[0,:,:,0] = cam.getFrame() / 255 - .5
                _, l1 = session.run([optimizer, loss], feed_dict={
                                       data_input: frame})
                print("loss",l,l1)
            for i in range(buffer_length):
                if temp_buffer:
                    frame[0,:,:,:] = buffer[i, :, :, :]
                    _, l = session.run([optimizer, loss], feed_dict={
                                       data_input: frame})
                    if l < FLAG['avg']:
                        buffer[i, :, :, :] = temp_buffer.popleft()
                else:
                    break
            print("Cleaned buffer up to",i,FLAG)

    except KeyboardInterrupt:
        pass

    finally:
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
