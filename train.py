"""
Training Service: Trains models on local data, then publishes new model.

Phase 1: Basic Operation
Just publishes a timestamp.
"""

# Basic version should just publish time stamp to the outbox.
import os
import time
import asyncio
import random
import datetime
# Import the learning and logging packages.
from learn import Model, Data
from simlog import Log
from aggregate import aggregate

# Number of epochs per training cycle.
NUM_EPOCHS = 10

# Paths.
DATA_PATH = os.path.join(os.getcwd(), "data", "client.data")
MODEL_PATH = os.path.join(os.getcwd(), "model.torch")
PUBLISHED_MODEL_PATH = os.path.join(os.getcwd(), "outbox", "model.torch")
AGG_MODEL_FN = "model.agg.torch"
AGG_MODEL_PATH = os.path.join(os.getcwd(), AGG_MODEL_FN)

# Logs.
log = Log("TRAIN", os.path.join(os.getcwd(), "logs", "train.log"))
# test_log = Log("TEST", os.path.join(os.getcwd(), "logs", "test.log"))

def main():
    log.log("Kicking off training...")
    round_counter = 0
    while True:
        round_counter += 1
        # Load the model.
        log.log("\tLoading model...")
        model = Model(MODEL_PATH, log=log)
        # Train the model.
        log.log(f"\tTraining model for {NUM_EPOCHS} epochs...")
        for i in range(1, NUM_EPOCHS+1):
            log.log(f"ROUND {round_counter}: EPOCH {i}")
            model.train(Data(DATA_PATH, log=log).train_dataloader)
            model.test(Data(DATA_PATH, log=log).test_dataloader)
            # Publish message to both the outbox and the current model.
        log.log("\tPublishing model update...")
        model.save(MODEL_PATH)
        model.save(PUBLISHED_MODEL_PATH)
        log.log("\tTraining done.\nPost-training results are below.")
        model.test(Data(DATA_PATH, log=log).test_dataloader)
        # Now wait for the an aggregated model to show up.
        while True:
            try:
                # Try to find aggregated model and generate model from it.
                # log.log("Searching for aggregated model...")
                model.load(AGG_MODEL_PATH)
                # If successful save the aggregated model
                log.log(f"\tAggregated model found!\n\t{AGG_MODEL_PATH} --> {MODEL_PATH}")
                model.save(MODEL_PATH)
                # Test the aggregated model.
                log.log(f"Post-aggregation testing results are below.")
                model.load(MODEL_PATH)
                model.test(Data(DATA_PATH, log=log).test_dataloader)
                break
            except FileNotFoundError:
                # Else wait a bit, then try again.
                time.sleep(10)
                continue


if __name__ == "__main__":
    main()