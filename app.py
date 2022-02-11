# A simple app.

import time
import socket

counter = 0
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
while True:
    time.sleep(5)
    counter += 5
    print(f"{hostname} @ {ip} :: {counter} seconds have passed.")