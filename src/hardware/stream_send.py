from __future__ import print_function
import lib.camera as camera
import socket
from time import sleep
from sys import stdout

cam = camera.Camera()
cam.setSize(160, 120)

address = ('192.168.1.4', 8089)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(address)
sock.listen(1)

while True:

    try:
        connection, peer_address = sock.accept()
        print("Connected to", peer_address)
        cam.send(connection)

    except socket.error:
        print("Socket error")
        continue
