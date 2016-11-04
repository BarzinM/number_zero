import cv2


def setWidthHeight(width, height, device_number=0):
    # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set
    cap = cv2.VideoCapture(device_number)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


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
    import pickle
    import struct

    cap = cv2.VideoCapture(device_number)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind((ip, port))
    sock.listen(1)
    connection, address = sock.accept()
    print('Recieved connection from',str(address))
    # stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pickle.dumps(frame)
            print('sending',len(data))
            connection.sendall(struct.pack("L", len(data)))
            while len(data)>0:
                connection.sendall(data[:1024])
                data = data[1024:]
    except Exception:
        raise
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\rReleased all resources!!!")


def streamRecieve(ip, port):
    import socket
    import struct
    import pickle

    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(10)
    connection.connect((ip,port))
    # connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    data_size = struct.calcsize("L")
    data = ""
    while True:
        # while len(data) < data_size:
        data = connection.recv(data_size)
        packed_message_size = data
        message_size = struct.unpack("L", packed_message_size)[0]
        # data = data[data_size:]
        frame_data = ""
        print(message_size)
        while len(frame_data) < message_size:
            frame_data+=connection.recv(100)
        # frame_data = connection.recv(message_size)
        # frame_data = data[:message_size]
        # data = data[message_size:]

        print('showing')
        try:
            frame = pickle.loads(frame_data)
        except EOFError:
            pass
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    
