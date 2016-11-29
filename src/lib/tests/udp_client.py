## CLIENT

from communication import UDP


udp = UDP('localhost',8089)

with udp:
    udp.send('first message')
    udp.send('secnod message')
    print(udp.receive())