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
# Import the learning and logging packages.
from learn import Model, Data
from logging import Log

def main():
    log = Log("TRAIN", os.path.join(os.getcwd(), "logs", "train.log"))
    log.log("Kicking off training...")
    while True:
        # Load the model.
        log.log("\tLoading model...")
        model = Model("model.h5")
        # Train the model.
        log.log("\tTraining model...")
        model.train(Data().training_data)
        # Publish message to both the outbox and the current model.
        log.log("\tPublishing model...")
        model_path = os.path.join(os.getcwd(), "outbox", "model.h5")
        model.save("model.h5")
        model.save(model_path)
        log.log("\tTraining done.")

if __name__ == "__main__":
    main()