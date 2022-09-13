import os
import time
import torch
import socket
import asyncio

from simlog import Log
from learn import Model, DataFashion, DataMNIST

from config import NODELIST_FN, MODEL_FN, MODEL_PATH, \
    AGG_MODEL_PATH, INBOX_PATH, DATA_PATH, AGGREGATION_QUOTA

# Get nodes.
with open(NODELIST_FN, 'r') as f:
     NODE_LIST = f.read().strip().split()
NODE_LIST.remove(socket.gethostbyname(socket.gethostname()))
# Path information
ALL_OTHER_MODELS = [os.path.join(INBOX_PATH, address, MODEL_FN) \
    for address in NODE_LIST]

def aggregate(log:Log=None):
    """
    This function takes a model (weights stored in model_path) and aggregates 
    it along with the received models (weights stored in the subdirectories in 
    inbox_path) and then returns the aggregate model (by storing the resulting 
    weights in output_path). Model filename is assumed to be model_fn.
    """
    # Set the model.
    model = Model(MODEL_PATH)
    # Set display output to either log or print.
    if log is not None:
        display = log.log
        model.display = log.log
    else:
        display = print
        model.display = print
    # COLLECTING MODELS.
    # List to store all model weights.
    all_weights = []
    # Check each path for a model.
    processed_model_paths = []
    for other_model in ALL_OTHER_MODELS:
        try:
            # display(f"Checking {other_model}...")
            all_weights.append(torch.load(other_model))
            processed_model_paths.append(other_model)
            display(f"\tModel retrieved at {other_model}.")
        except FileNotFoundError as e:
            pass
            # display(f"\tModel could not be found.")
    # If the quota of models is met, proceed, else skip aggregation this time.
    if len(processed_model_paths) < AGGREGATION_QUOTA:
        return False
    # AGGREGATION.
    # Clean up (remove) the processed models.
    log.log("\nOther model information:\n")
    for processed_model in processed_model_paths:
        # Test the model. (Should be removed later)
        other_model = Model(processed_model, log=log)
        # log.log(next(iter(other_model.model.state_dict()))[1])
        other_model.test(DataFashion(DATA_PATH, log=log).test_dataloader)
        # Remove model.
        log.log(f"Model at {processed_model} cleared.")
        os.remove(processed_model)
    # Test home model. (Should be removed later)
    log.log("\nHome model information:\n")
    # log.log(next(iter(model.model.state_dict()))[0])
    model.test(DataFashion(DATA_PATH, log=log).test_dataloader)
    # Aggregate through the home model.
    model.aggregate(all_weights)
    # Save the model.
    model.save(AGG_MODEL_PATH)
    # Indicate success.
    return True

def main():
    log = Log("AGGREGATE", os.path.join(os.getcwd(), "logs", "agg.log"))
    # Periodically execute aggregation process.
    log.log(f"{ALL_OTHER_MODELS}")
    # time.sleep(10)
    while True:
        success = aggregate(log=log)
        if success:
            log.log("Model aggregated successfully!\n")
            time.sleep(10)
        else:
            time.sleep(10)

if __name__ == "__main__":
    main()