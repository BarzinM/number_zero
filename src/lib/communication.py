from __future__ import print_function
import socket
import struct


def getOwnIP():
    # Get the IP address of the computer executing this file
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    temp_sock.connect(('192.168.0.1', 0))
    ip_address = temp_sock.getsockname()[0]
    temp_sock.close()
    return ip_address


class TCP(object):
    def __init__(self, server_port, server_ip=None):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if server_ip is None:
            self.socket.bind((getOwnIP(), server_port))
            self.socket.listen(1)
            self.connection, address = self.socket.accept()
        else:
            self.socket.connect((server_ip, server_port))
            self.connection = self.socket

    def send(self, message):
        self.connection.sendall("sp")
        self.connection.sendall(struct.pack(">L", len(message)))
        self.connection.sendall(struct.pack(">L", len(message)))
        for i in range(0, len(message), 4096):
            self.connection.sendall(message[i:i+4096])
        self.connection.sendall("ep")

    def receive(self):
        data_size = struct.calcsize(">L")
        offset = 2* data_size + 2
        data = self.buffer
        while True:
            pointer = -1
            while len(data) < 4096:
                data += connection.recv(4096)
            for i in range(len(data)-offset):
                if data[i]=='s' and data[i+1]=='p':
                    pointer = i +2
                    break
            if pointer < 0:
                data = data[-offset:] + connection.recv(4096)
                continue
        message_size = struct.unpack(
            ">L", data[pointer:pointer + data_size])[0]
        pointer += data_size
        message_size_2 = struct.unpack(
            ">L", data[pointer:pointer + data_size])[0]
        pointer += data_size
        if message_size != message_size_2:
            print 'bad packet size info'
            continue

        while len(data) < message_size + pointer:
            data += connection.recv(4096)

        pointer += message_size

        if data[pointer:pointer + 2] != "ep":
            print("bad end")
            self.buffer = data[pointer:]
            continue

        self.buffer = data[pointer + 2:]
        return data[pointer:pointer + message_size]

class UDP(object):
    def __init__(self, server_port, server_ip=None):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if server_ip is None:
            server_ip = getOwnIP()
        self.address = (server_ip, server_port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.socket.close()

    def send(self, message):
        return self.socket.sendto(message, self.address)

    def respond(self, message):
        return self.socket.sendto(message, self.peer_address)

    def receive(self, size=4096):
        data, self.peer_address = self.socket.recvfrom(4096)

if __name__ == "__main__":
    print('This device IP is', getOwnIP())
