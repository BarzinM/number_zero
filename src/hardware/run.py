import lib.camera as camera
from hardware.motor_control import Motor
import socket
import struct
from time import sleep
from sys import stdout

# cam = camera.Camera()
# cam.setSize(160, 120)

motor_1 = Motor("P9.14", 'pwm0',48,60)
motor_2 = Motor("P9.16", 'pwm1',49,115)

address = ('192.168.1.4', 8089)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(address)
sock.listen(1)


message_format = ">cffc"
message_size = struct.calcsize(message_format)

while True:

    try:
        connection, peer_address = sock.accept()

        # cam.send(connection)
        while True:
            message = connection.recv(message_size)
            _, forward, turn, _ = struct.unpack(message_format, message)

            print("%+ 1.3f | %+ 1.3f" % (forward, turn))
            stdout.write("\033[F\033[K")

            motor_1.setValue(forward)
            motor_2.setValue(turn)
    except socket.error:
        continue
