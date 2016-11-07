from __future__ import print_function
import socket


def getOwnIP():
    # Get the IP address of the computer executing this file
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    temp_sock.connect(('192.168.0.1', 0))
    ip_address = temp_sock.getsockname()[0]
    temp_sock.close()
    return ip_address    

class TCP(object):
    def __init__(self):
        pass

class UDP(object):
    def __init__(self, server_port,server_ip=None):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (server_ip,server_port)
        if server is True:
            self.socket.bind(self.address) 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.socket.close()

    def send(self, message):
        return self.socket.sendto(message, self.address)
        
    def respond(self,message):
        return self.socket.sendto(message,self.peer_address)

    def receive(self,size=4096):
        data, self.peer_address = self.socket.recvfrom(4096)

if __name__ == "__main__":
    print('This device IP is',getOwnIP())