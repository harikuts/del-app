"""
Transmission Service: Sends out messages to other connected nodes.

Phase 1: Basic Operation
It should just send a generic broadcast message.
"""

# Globals
NODELIST_FN = "nodelist.txt"
PORT = 1245

import time
import socket
import random
import os

CUR_DIR = os.getcwd()
OUTBOX_PATH = os.path.join(CUR_DIR, "outbox")
MESSAGE_PATH = os.path.join(OUTBOX_PATH, "message.txt")

# Get hostnames
with open(NODELIST_FN, 'r') as f:
    nodelist = f.read().strip().split()

# Messaging loop.
while True:
    for host in nodelist:
        time.sleep(random.randint(0, 5))
        # Get message from outbox.
        try:
            with open(MESSAGE_PATH, 'r') as f:
                message = f.read()
        except:
            continue
        # Open socket for sending message.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, PORT))
            s.send(message.encode("utf-8"))
        except Exception as e:
            pass
            print(f"Failed to connect to {host}. Skipping.")
        s.close()