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
from simlog import Log

# Number of epochs per training cycle.
NUM_EPOCHS = 5

DATA_PATH = "data/test_client.data"
MODEL_PATH = os.path.join(os.getcwd(), "model.torch")
OUT_MODEL_PATH = os.path.join(os.getcwd(), "outbox", "model.torch")
log = Log("TRAIN", os.path.join(os.getcwd(), "logs", "train.log"))

def main():
    log.log("Kicking off training...")
    while True:
        # Load the model.
        log.log("\tLoading model...")
        model = Model(MODEL_PATH, log=log)
        # Train the model.
        log.log(f"\tTraining model for {NUM_EPOCHS} epochs...")
        for i in range(1, NUM_EPOCHS+1):
            model.train(Data(DATA_PATH, log=log).train_dataloader)
            # Publish message to both the outbox and the current model.
            log.log("\tPublishing model update...")
            model.save(MODEL_PATH)
            model.save(OUT_MODEL_PATH)
        log.log("\tTraining done.")
        time.sleep(5)

if __name__ == "__main__":
    main()