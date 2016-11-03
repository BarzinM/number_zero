import socket


class UDP(object):
    def __init__(self, server_ip, server_port, server=False):
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