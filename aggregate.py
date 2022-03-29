import os
import time
import torch

from simlog import Log
from learn import Model

# File I/O information.
NODELIST_FN = "nodelist.txt"
with open(NODELIST_FN, 'r') as f:
        NODE_LIST = f.read().strip().split()
STORED_FN = "model.torch"
CUR_DIR = os.getcwd()
THIS_MODEL = os.path.join(CUR_DIR, STORED_FN)
INBOX_PATH = os.path.join(CUR_DIR, "inbox")
ALL_OTHER_MODELS = [os.path.join(INBOX_PATH, address, STORED_FN) \
    for address in NODE_LIST]

def aggregate(model:Model, inbox_path:str=INBOX_PATH, model_fn:str=STORED_FN, output_path:str=THIS_MODEL, log:Log=None):
    """
    This function takes a model (weights stored in model_path) and aggregates 
    it along with the received models (weights stored in the subdirectories in 
    inbox_path) and then returns the aggregate model (by storing the resulting 
    weights in output_path). Model filename is assumed to be model_fn.
    """
    # Set display output to either log or print.
    if log is not None:
        display = log.log
    else:
        display = print
    # List to store all model weights.
    all_weights = []
    # Store paths to all other model paths.
    all_other_models = [os.path.join(inbox_path, address, model_fn) \
        for address in NODE_LIST]
    # Check each path for a model.
    for other_model in all_other_models:
        try:
            display(f"Checking {other_model}...")
            all_weights.append(torch.load(other_model))
            os.remove(other_model)
            display(f"\tModel retrieved and cleared..")
        except Exception as e:
            display(f"\tModel could not be found.")
    # Aggregate through the home model.
    model.aggregate(all_weights)
    # Save the model.
    model.save(output_path)
    # Return the model, in case it's desired.
    return model

def main():
    # Periodically execute aggregation process.
    pass

if __name__ == "__main__":
    main()