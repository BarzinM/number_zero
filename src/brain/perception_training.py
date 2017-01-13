# dependenices
from __future__ import print_function
import collections
from perception import Encoder
import numpy as np
from lib.camera import Camera
import socket
import tensorflow as tf
from preprocessing import showMultipleArraysHorizontally
import threading
from time import sleep, time
import matplotlib.pyplot as pp


hardcore_train_steps = 10
buffer_length = 100
temp_buffer_length = 10
batch_size = 16
height = 120
width = 160
remote_camera = False
body_address = ('192.168.1.4', 8089)
average_running_ratio = .1
initial_delay = 1

FLAG = {'time_point': 1.}
FLAG['running'] = True

# initialize model

data_input = tf.placeholder(tf.float32, shape=(
    None, height, width, 1), name="data_input")
net = Encoder(data_input)
output = net.decoded

loss = tf.reduce_mean(tf.abs(output - data_input))
optimizer = tf.train.AdamOptimizer(.001).minimize(loss)


# initialize a numpy buffer
buffer_scores = np.zeros(buffer_length)
buffer = np.zeros((buffer_length, height, width, 1))

frame = np.empty((1, height, width, 1), np.float32)

temp_buffer = collections.deque(maxlen=temp_buffer_length)

# cmara setup
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

session = tf.Session()

fig = pp.figure()


def capture(session):
    ax = fig.gca()
    fig.show()
    sleep(initial_delay)

    frame = np.empty((1, height, width, 1), np.float32)
    time_point = time()
    max_cost = 0.

    while True:
        # run through encoder and calculate loss, calculate running average of
        # stream
        frame[0, :, :, 0] = cam.getFrame() / 255

        _, loss_value, decoded_output = session.run([optimizer, loss, output], feed_dict={
            data_input: frame})

        ax.imshow((decoded_output[0, :, :, 0]) * 255, cmap='gray')
        fig.canvas.draw()

        if loss_value > max_cost:
            max_cost = loss_value
            max_frame = frame

        if time() > time_point:
            time_point += FLAG['time_point']
            max_cost = 0
            temp_buffer.append(max_frame)

        sleep(.2)


def train(session, buffer, camera):
    frame = np.empty((1, height, width, 1), np.float32)

    sleep(initial_delay)
    previous_time = time()

    average_cost = 0.
    while FLAG['running']:
        print("=" * 20)

        i = 0
        while temp_buffer:
            temp = temp_buffer.popleft()
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: temp})
            if loss_value > buffer_scores[i]:
                buffer[i, :, :, :] = temp
                i += 1
        if i:
            print("Filled up to %i with cost >= %f" % (i, loss_value))

        for _ in range(hardcore_train_steps):
            indexes = np.random.randint(buffer_length, size=batch_size)
            batch = buffer[indexes, :, :, :]
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: batch})

            average_cost = average_running_ratio * loss_value + \
                (1. - average_running_ratio) * average_cost

        for i in range(buffer_length):
            frame[0, :, :, :] = buffer[i, :, :, :]
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: frame})
            buffer_scores[i] = loss_value

        sorted_index = np.argsort(buffer_scores)
        buffer = buffer[sorted_index]
        print("test:",np.sum(buffer[buffer_length-2,:,:,:]-buffer[buffer_length-1,:,:,:]))

        print("Average cost is", average_cost)

        now_time = time()
        duration = now_time - previous_time
        previous_time = now_time
        FLAG['time_point'] = duration / temp_buffer_length


with session.as_default():
    tf.initialize_all_variables().run()

    try:
        start_time = time()
        frame[0, :, :, 0] = cam.getFrame() / 255
        _ = session.run([output], feed_dict={
            data_input: frame})
        end_time = time()
        print("Forward propogation took %f seconds" % (end_time - start_time))

        start_time = time()
        frame[0, :, :, 0] = cam.getFrame() / 255
        _ = session.run([loss], feed_dict={
            data_input: frame})
        end_time = time()
        print("Loss calculation took %f seconds" % (end_time - start_time))

        start_time = time()
        frame[0, :, :, 0] = cam.getFrame() / 255
        _ = session.run([loss, output], feed_dict={
            data_input: frame})
        end_time = time()
        print("Forward propogation and loss calculation together took %f seconds" % (
            end_time - start_time))

        start_time = time()
        for i in range(buffer_length):
            frame[0, :, :, 0] = cam.getFrame() / 255
            session.run([optimizer], feed_dict={
                data_input: frame})
            buffer[i, :, :, :] = frame

        print("finished initialization in:", time() - start_time)


        thrd = threading.Thread(target=train, args=(session, buffer, cam))
        thrd.daemon = True
        thrd.start()

        capture(session)

    except KeyboardInterrupt:
        pass

    finally:
        FLAG['running'] = False
        pp.close(fig)
        cam.close()
        fig = pp.figure()
        number_of_images = 20
        rows = np.floor(number_of_images / 4) + 1
        columns = min(number_of_images, 4)
        for i in range(number_of_images):
            ax = fig.add_subplot(rows, columns, i + 1)
            pp.imshow(buffer[buffer_length - i*4-1,:,:,0], cmap=pp.get_cmap('gray'))
            pp.axis('off')
        pp.show()
        # showMultipleArraysHorizontally(
        #     buffer[-20:, :, :, 0], max_per_row=4)
        thrd.join()
        print("Train thread joined")
