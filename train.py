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
# Import the learning package.
from learn import Model, Data

def main():
    while True:
        
        # Load the model.
        model = Model("model.h5")
        # Train the model.
        model.train(Data().training_data)
        # Publish message to both the outbox and the current model.
        print("Publishing model.")
        model_path = os.path.join(os.getcwd(), "outbox", "model.h5")
        model.save("model.h5")
        model.save(model_path)

if __name__ == "__main__":
    main()