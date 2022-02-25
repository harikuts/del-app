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

from logging import Log

CUR_DIR = os.getcwd()
OUTBOX_PATH = os.path.join(CUR_DIR, "outbox")
MESSAGE_PATH = os.path.join(OUTBOX_PATH, "model.h5")

# Get hostnames
with open(NODELIST_FN, 'r') as f:
    nodelist = f.read().strip().split()

def main():
    log = Log("TX", os.path.join(os.getcwd(), "logs", "tx.log"))
    # Messaging loop.
    while True:
        time.sleep(random.randint(0, 5))
        log.log("Starting transmission...")
        # Get message from outbox.
        log.log(f"\tRetrieving model...")
        try:
            with open(MESSAGE_PATH, 'rb') as f:
                message = f.read()
            log.log(f"\t\tRetrieved.")
        except:
            continue
        # Send to each connected host.
        log.log("\tSending to connected hosts...")
        for host in nodelist:
            # Open socket for sending message.
            log.log(f"\t\t--> {host}")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((host, PORT))
                s.send(message)
                log.log(f"\t\t\tSent.")
            except Exception as e:
                pass
                log.log(f"\t\t\tFailed to connect to {host}. Skipping.")
            s.close()

if __name__ == "__main__":
    main()