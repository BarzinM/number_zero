## SERVER
from communication import UDP

udp=UDP('localhost',8089,True)

with udp:
    while True:
        data=''
        data = udp.receive()
        print(data)

        if data:
            udp.respond('This is the response')