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
from sys import stdout
from number_stream import RunningAverage
import os.path


hardcore_train_steps = 10
buffer_length = 100
temp_buffer_length = 20
batch_size = 16
remote_camera = False
initial_delay = 0
save = False
load = False
saved_directory = './saved'
body_address = ('192.168.1.4', 8089)
height = 120
width = 160
randomized_input = False

assert 2 * temp_buffer_length <= buffer_length, "Size of temp_buffer_length is too large!"

FLAG = {'time_point': 1., 'running': True, 'buffer_avg': 0., 'stream_avg': 0.}

# cmara setup
cam = Camera()

if remote_camera:
    # connect to body
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(100)
    connection.connect(body_address)

    # receive stream
    cam.receive(connection)
else:
    cam.setSize(width, height)
    cam.capture()

image_sample = cam.getFrame()
if len(image_sample.shape) == 2:
    image_depth = 1
else:
    image_depth = 3
print("Camera image depth is", image_depth)

# initialize model
data_input = tf.placeholder(tf.float32, shape=(
    None, height, width, image_depth), name="data_input")
net = Encoder([height, width, image_depth])
if randomized_input:
    randomized = data_input + tf.random_normal([1, 120, 160, 1], stddev=.05)
    output = net.model(randomized)
else:
    output = net.model(data_input)

loss = tf.reduce_mean(tf.square(output - data_input))
optimizer = tf.train.AdamOptimizer(.001).minimize(loss)


# initialize memory
buffer_scores = np.ones(buffer_length)
temp_buffer = collections.deque(maxlen=temp_buffer_length)
buffer = np.zeros((buffer_length, height, width, image_depth))

if load and os.path.isfile(os.path.join(saved_directory, 'buffer.npy')):
    saved_buffer = np.load(os.path.join(saved_directory, 'buffer.npy'))
    saved_buffer_length = saved_buffer.shape[0]
    buffer[:saved_buffer_length, :, :, :] = saved_buffer
    del saved_buffer
    print("Loaded %i of buffer from saved data:" %
          saved_buffer_length, os.path.join(saved_directory, 'buffer.npy'))
else:
    saved_buffer_length = 0

for i in range(saved_buffer_length, buffer_length):
    buffer[i:i + 1, :, :, 0] = cam.getFrame() / 255
    sleep(.05)
print("Initialized %i elements of buffer from camera." %
      (buffer_length - saved_buffer_length))


session = tf.Session()

fig = pp.figure()


def capture(session):
    ax = fig.gca()
    fig.show()
    pp.axis('off')
    sleep(initial_delay)

    time_point = time()
    max_cost = 0.
    previous_frame = cam.getFrame()

    threshold = 1000.

    while True:
        # run through encoder and calculate loss, calculate running average of
        # stream
        print("Buffer: %.3f | Stream: %.3f" %
              (FLAG['buffer_avg'], FLAG['stream_avg']))
        stdout.write("\033[F\033[K")

        sleep(.2)

        frame = cam.getFrame() / 255

        loss_value, decoded_output = session.run([loss, output], feed_dict={
            data_input: frame[None, :, :, None]})
        FLAG['stream_avg'] = loss_value
        ax.imshow(decoded_output[0, :, :, 0] * 255, cmap='gray')
        fig.canvas.draw()

        diff = np.sum(np.abs(frame-previous_frame))
        if diff<threshold or loss_value<FLAG['buffer_avg']:
            continue

        _,loss_value = session.run([optimizer,loss], feed_dict={
            data_input: frame[None, :, :, None]})

        previous_frame[:] = frame

        if loss_value > max_cost:
            max_cost = loss_value
            max_frame = frame

        if time() > time_point:
            time_point += FLAG['time_point']
            max_cost = 0.
            temp_buffer.append(max_frame[None, :, :, None])



def train(session, camera):
    sleep(initial_delay)
    previous_time = time()
    avg=RunningAverage(initial=1.)

    print("=" * 40)
    while FLAG['running']:

        #for _ in range(hardcore_train_steps):
        while len(temp_buffer)<temp_buffer_length:
            if not FLAG['running']:  # remove for deployment
                break
            indexes = np.random.randint(buffer_length, size=batch_size)
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: buffer[indexes, :, :, :]})

            FLAG['buffer_avg'] = avg.average(loss_value)

        print("finished hard training")

        for i in range(buffer_length):
            if not FLAG['running']:  # remove for deployment
                break
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: buffer[i:i + 1, :, :, :]})
            buffer_scores[i] = loss_value

        sorted_index = np.argsort(buffer_scores)
        buffer[:] = buffer[sorted_index]

        now_time = time()
        duration = now_time - previous_time
        previous_time = now_time
        FLAG['time_point'] = duration / temp_buffer_length

        print("finished sorting")

        i = 0
        while temp_buffer and i< temp_buffer_length:
            temp = temp_buffer.popleft()
            _, loss_value = session.run([optimizer, loss], feed_dict={
                data_input: temp})
            if loss_value > buffer_scores[2*i]:
                buffer[2*i, :, :, :] = temp
                i += 1
        if i:
            print("=" * 20)
            print("Added %i to buffer with cost >= %f" % (i - 1, loss_value))

with session.as_default():
    tf.initialize_all_variables().run()
    if load:
        net.load(session, os.path.join(saved_directory, 'perception-model'))

    try:
        print("=" * 20)
        print("Performance analysis:")
        start_time = time()
        frame = cam.getFrame() / 255
        _ = session.run([output], feed_dict={
            data_input: frame[None, :, :, None]})
        end_time = time()
        print("Forward propogation took %f seconds" % (end_time - start_time))

        start_time = time()
        frame = cam.getFrame() / 255
        _ = session.run([loss], feed_dict={
            data_input: frame[None, :, :, None]})
        end_time = time()
        print("Loss calculation took %f seconds" % (end_time - start_time))

        start_time = time()
        frame = cam.getFrame() / 255
        _ = session.run([optimizer], feed_dict={
            data_input: frame[None, :, :, None]})
        end_time = time()
        print("Optimization took %f seconds" % (end_time - start_time))

        thrd = threading.Thread(target=train, args=(session, cam))
        thrd.daemon = True
        thrd.start()

        capture(session)

    except KeyboardInterrupt:
        pass
        if save:
            net.save(session, os.path.join(
                saved_directory, 'perception-model'))
            np.save(os.path.join(saved_directory, 'buffer.npy'), buffer)

    finally:
        print("\nPlease wait for finalization of the process ...")
        FLAG['running'] = False
        pp.close(fig)
        cam.close()
        thrd.join()
        showMultipleArraysHorizontally(
            buffer[-20:, :, :, 0], max_per_row=4)
        print("Train thread joined. Process finished.")
