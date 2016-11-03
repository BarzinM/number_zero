import cv2


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
    import pickle
    import struct

    cap = cv2.VideoCapture(device_number)
    stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # stream.connect((ip, port))
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pickle.dumps(frame)
            stream.sendall(struct.pack("L", len(data)) + data)
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise e


def streamRecieve(port):
    import socket
    import struct
    import pickle

    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connection.bind(('localhost', port))
    # connection.listen(1)
    # stream, address = connection.accept()
    print("Connection Established with",address)
    data = ""
    data_size = struct.calcsize("L")
    while True:
        while len(data) < data_size:
            data += connection.recvfrom(4096)
        packed_message_size = data[:data_size]
        data = data[data_size:]
        message_size = struct.unpack("L", packed_message_size)[0]
        while len(data)<message_size:
            data += connection.recvfrom(4096)
        frame_data = data[:message_size]
        data = data[message_size:]

        frame = pickle.loads(frame_data)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
