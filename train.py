"""
Training Service: Trains models on local data, then publishes new model.

Phase 1: Basic Operation
Just publishes a timestamp.
"""

# Basic version should just publish time stamp to the outbox.
import os
import time
import random
import datetime

def main():
    while True:
        time.sleep(random.randint(5, 7))
        # Publish message
        print("Publishing message.")
        message = f"Hello. -- {datetime.datetime.now()}\n"
        message_path = os.path.join(os.getcwd(), "outbox", "message.txt")
        with open(message_path, 'w') as f:
            f.write(message)

if __name__ == "__main__":
    main()