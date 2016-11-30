from __future__ import print_function
import cv2
import threading
from time import sleep


class Camera(object):
    def __init__(self):
        self.width = None
        self.height = None
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        pass

    def _applySize(self):
        if self.width is not None and self.height is not None:
            self.cap.set(3, self.width)
            self.cap.set(4, self.height)

    def setSize(self, width, height):
        self.width = width
        self.height = height

    def send(self, ip, port, device_number=0):
        pass

    def receive(self, ip, port):
        pass

    def _capture(self):
        cap = self.cap
        while self.running:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            with self.frame_lock:
                self.frame = gray

        cap.release()

    def capture(self, device_number=0):
        self.cap = cv2.VideoCapture(device_number)
        self._applySize()
        thrd = threading.Thread(target=self._capture)
        thrd.daemon = True
        thrd.start()
        while self.frame is None:
            sleep(.01)

    def monitor(self, device_number=0):
        cap = cv2.VideoCapture(device_number)

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display(self):
        while True:
            cv2.imshow('frame', self.getFrame())
            cv2.waitKey(1)

    def getFrame(self):
        with self.frame_lock:
            return self.frame


def setWidthHeight(width, height, device_number=0):
    # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set
    cap = cv2.VideoCapture(device_number)
    cap.set(3, width)
    cap.set(4, height)


def setFPS(fps, device_number=0):
    cap = cv2.VideoCapture(device_number)
    cap.set(cv2.CAP_PROP_FPS, fps)


def monitor(device_number=0):

    cap = cv2.VideoCapture(device_number)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def streamSend(ip, port, device_number=0):
    import socket
    import cPickle as pickle
    import struct

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.listen(1)
    connection, address = sock.accept()
    print('Recieved connection from ' + str(address))

    data_size = struct.calcsize(">L")

    cap = cv2.VideoCapture(device_number)
    cap.set(3, 160)
    cap.set(4, 120)

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pickle.dumps(frame)

            connection.sendall("snp")
            connection.sendall(struct.pack(">L", len(data)))
            connection.sendall(struct.pack(">L", len(data)))
            for i in range(0, len(data), 4096):
                connection.sendall(data[i:i + 4096])
            connection.sendall("ep")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\rReleased all resources!!!")


def streamToFile(ip, port, directory, data_count):
    import socket
    import struct
    import cPickle as pickle
    import numpy as np
    import glob
    import os

    def arrayToFile(file_name, array):
        file_handle = open(file_name, "wb")
        number_of_data = array.shape[0]
        print('Saving %d data samples into "%s" ...' %
              (number_of_data, file_name))

        np.save(file_handle, array)

        # always close the file
        file_handle.close()

    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(100)
    connection.connect((ip, port))

    data_size = struct.calcsize(">L")
    offset = 2 * data_size + 3

    dataset = np.empty((data_count, 120, 160), np.float32)
    filled_index = 0
    ls = glob.glob(os.path.join(directory, '*.npdata'))
    print(ls)
    file_count = len(ls)
    file_name = os.path.join(directory, str(file_count) + '.npdata')

    data = connection.recv(4096)

    while True:
        try:
            pointer = -1
            while len(data) < 4096:
                data += connection.recv(4096)

            for i in range(len(data) - offset):
                if data[i] == 's' and data[i + 1] == 'n' and data[i + 2] == 'p':
                    pointer = i + 3
                    break
            if pointer < 0:
                data = data[-offset:] + connection.recv(4096)
                print('looking again', len(data))
                continue

            message_size = struct.unpack(
                ">L", data[pointer:pointer + data_size])[0]
            pointer += data_size
            message_size_2 = struct.unpack(
                ">L", data[pointer:pointer + data_size])[0]
            pointer += data_size
            if message_size != message_size_2:
                print('bad packet size info')
                continue

            while len(data) < message_size + pointer:
                data += connection.recv(4096)

            frame_data = data[pointer:pointer + message_size]
            pointer += message_size

            if data[pointer:pointer + 2] != "ep":
                print("bad end")
                data = data[pointer:]
                continue

            data = data[pointer + 2:]

            frame = pickle.loads(frame_data)
            dataset[filled_index, :, :] = frame
            filled_index += 1
            print("Received %i images" %((file_count*data_count)+filled_index))

            if filled_index + 1 == data_count:
                arrayToFile(file_name, dataset)
                dataset = np.empty((data_count, 120, 160), np.float32)
                filled_index = 0
                file_count += 1
                file_name = os.path.join(
                    directory, str(file_count) + '.npdata')
        except KeyboardInterrupt:
            break

        except socket.timeout:
            print("Timeout happened. dumping buffer and trying again.")
            continue


def streamRecieve(ip, port):
    import socket
    import struct
    import cPickle as pickle

    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(100)
    connection.connect((ip, port))

    data_size = struct.calcsize(">L")
    offset = 2 * data_size + 3

    data = connection.recv(4096)
    frame_recived = 0
    cv2.namedWindow('Stream from remote', cv2.WINDOW_NORMAL)

    while True:
        pointer = -1
        while len(data) < 4096:
            data += connection.recv(4096)

        for i in range(len(data) - offset):
            if data[i] == 's' and data[i + 1] == 'n' and data[i + 2] == 'p':
                pointer = i + 3
                break
        if pointer < 0:
            data = data[-offset:] + connection.recv(4096)
            print('looking again', len(data))
            continue

        message_size = struct.unpack(
            ">L", data[pointer:pointer + data_size])[0]
        pointer += data_size
        message_size_2 = struct.unpack(
            ">L", data[pointer:pointer + data_size])[0]
        pointer += data_size
        if message_size != message_size_2:
            print('bad packet size info')
            continue

        while len(data) < message_size + pointer:
            data += connection.recv(4096)

        frame_data = data[pointer:pointer + message_size]
        pointer += message_size

        if data[pointer:pointer + 2] != "ep":
            print("bad end")
            data = data[pointer:]
            continue

        data = data[pointer + 2:]

        frame = pickle.loads(frame_data)

        frame_recived += 1
        cv2.putText(frame, '# %d' % frame_recived, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Stream from remote', frame)
        cv2.waitKey(1)


def streamSendNewImages(ip, port, device_number=0):
    import socket
    import cPickle as pickle
    import struct

    cap = cv2.VideoCapture(device_number)
    cap.set(3, 160)
    cap.set(4, 120)

    ret, frame = cap.read()
    frame_previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.listen(1)
    connection, address = sock.accept()
    print('Recieved connection from', str(address))

    data_size = struct.calcsize(">L")
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(frame, frame_previous)

            if diff.mean() < 20:  # threshold for recognizing change in frames
                continue
            frame_previous = frame

            data = pickle.dumps(frame)

            connection.sendall("snp")
            data_length = len(data)
            connection.sendall(struct.pack(">L", data_length))
            connection.sendall(struct.pack(">L", data_length))
            for i in range(0, data_length, 4096):
                connection.sendall(data[i:i + 4096])
            connection.sendall("ep")

    except (KeyboardInterrupt, socket.error):
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\rReleased all resources!!!")


if __name__ == "__main__":
    cam = Camera()
    cam.capture(0)
    cam.display()
    
