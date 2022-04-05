"""
Receiver Service: Accepts messages from other nodes and saves them.

Phase 1: Basic Operation
It should just accept incoming messages. Kinda like a server.
"""

import socket
from _thread import *
import threading
import os

from simlog import Log

from config import PORT, DATASIZE, NODELIST_FN, MODEL_FN, INBOX_PATH

rx_lock = threading.Lock()

def rx_thread(conn, address, log):
    # Get stream of data.
    data_exists = False
    full_data = bytes()
    while True:
        data = conn.recv(DATASIZE)
        # Break if no more data is streaming.
        if not data:
            rx_lock.release()
            break
        else:
            full_data += data
            data_exists = True
    # If there is data, print and store it.
    if data_exists:
        log.log(f"({address[0]}) {(full_data[:5])}...")
        # Store in the inbox.
        store_path = os.path.join(INBOX_PATH, address[0], MODEL_FN)
        with open(store_path, 'wb') as f:
            f.write(full_data)
        log.log(f"--> {store_path}")
    conn.close()

def main():
    log = Log("RX", os.path.join(os.getcwd(), "logs", "rx.log"))
    # Get information from nodelist.
    with open(NODELIST_FN, 'r') as f:
        nodelist = f.read().strip().split()

    # Get network interface information.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    # Choose arbitary port.
    port = PORT

    # Bind, listen, and connect to the port.
    s.bind((host, port))


    s.listen(len(nodelist))
    # Looping listening service.
    while True:
        # Accept a connection.
        conn, address = s.accept()
        rx_lock.acquire()
        log.log(f"Connected to {address[0]}!")
        # Dispatch the thread.
        start_new_thread(rx_thread, (conn, address, log,))
    s.close()

        

if __name__ == "__main__":
    main()