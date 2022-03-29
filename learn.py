"""
Learning Library:
Classes and methods that pertain to the learning process.
"""
import sys
import pickle
from typing import OrderedDict
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
# import tensorflow

import simlog

"""
Global Variables:
Define any global variables needed by your scenarios.
"""

# All scenarios.
TRAIN_TEST_VAL_SPLIT = (0.6, 0.2, 0.2)
NUM_EPOCHS = 10 # Per training step.
BATCH_SIZE = 128

# # Test Twitter-LSTM Scenario
# SEQ_LEN = 2 # Number of preceeding words before next word.
# DATA_FILE = "data/test_data.txt"
# ENCODER_FILE = "data/encoder.pkl"
# ENCODER = pickle.load(open(ENCODER_FILE, 'rb'))
# VOCAB_SIZE = len(ENCODER) # Number of total words, which is mapped by the encoder.

"""
Model Library:
In this section, you can create models to use in your selector functions.
"""
def tf_LSTM():
    # Variables
    vocab_size = VOCAB_SIZE
    # Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(1024, input_shape=(SEQ_LEN, vocab_size), return_sequences=True))
    model.add(tf.keras.layers.LSTM(1024))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    return model

"""
Data Library:
In this section, you can create data to use in your selector functions.
Data should be passed out as X and Y.
"""

def TweetData(data_file, encoder):
    # Data file should contain the raw text content.
    # Encoder should be a pre-compiled tokenizer.

    # Load corpus from file.
    corpus = []
    with open(data_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            words = line.split()
            if len(words):
                corpus.append(words)

    # Build sequences of words.
    sequences = []
    for phrase in corpus:
        # Make sure the phrase can at least hold a sequence.
        if len(phrase) >= SEQ_LEN+1:
            # Start from the end of the first sequence and go to the end.
            for i in range (SEQ_LEN, len(phrase)):
                # Add the subset of words as a sequence.
                sequence = phrase[i-SEQ_LEN:i+1]
                sequences.append(sequence)

    # One-hot encoding to prepare dataset, starting with creating empty arrays.
    X = np.zeros((len(sequences), SEQ_LEN, VOCAB_SIZE), dtype=bool) # Holds potentially 3D array previous words.
    y = np.zeros((len(sequences), VOCAB_SIZE), dtype=bool) # Holds 2D array of next words.
    # Assemble encoded previous words and next words.
    for i, sequence in enumerate(sequences):
        # Grab previous words and next words.
        prev_words = sequence[:-1]
        next_word = sequence[-1]
        # Do the actual encoding (each row represents a next word or a set of previous words).
        for j, prev_word in enumerate(prev_words):
            X[i, j, encoder[prev_word]] = 1
        y[i, encoder[next_word]] = 1
    
    # Return X and y.
    return X, y



"""
Selector Classes:
These selector functions will point to the model and data you want to use.
You can create these functions in the model and data libraries below and
point those functions here.
"""

# PYTORCH MODEL AND DATA.

# Model class to interface with the application services.
class torch_Model():
    # Model: Basic 4-layer neural network for processing 28x28 input data.
    class SimpleModel_MNIST(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
            super(torch_Model.SimpleModel_MNIST, self).__init__()
            self.input_size, self.output_size = input_size, output_size
            # Create a simple 4-layer neural network with dense layers.
            self.nn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
        def forward(self, x):
            x = torch.reshape(x, (-1, self.input_size))
            output = self.nn(x)
            return output
    # Initialization function.
    def __init__(self, load_path:str=None, log:simlog.Log=None):
        # Set display output to either log or print.
        if log is not None:
            self.display = log.log
        else:
            self.display = print
        # Instantiate the model.
        self.model = self.SimpleModel_MNIST(28*28, 256, 10)
        if load_path is not None:
            # If load file is provided, load model from there.
            self.load(load_path)
        # Loss function and optimizer variables.
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    # Training function.
    def train(self, data:DataLoader):
        size = len(data.dataset)
        for i, (X, y) in enumerate(data):
            # X, y = X.to_device(), y.to_device()
            # Get prediction.
            pred = self.model(X)
            # Compute loss.
            loss = self.loss_function(pred, y)
            # Back propagation.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Print progress updates at every 10%.
            if i % (len(data)//10) == 0:
                loss, current = loss.item(), i * len(X)
                self.display(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    # Testing function.
    def test(self, data:DataLoader):
        size = len(data.dataset)
        num_batches = len(data)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data:
                # X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += self.loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        self.display(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg. loss: {test_loss:>8f} \n")
        return test_loss, correct
    # Save the model weights.
    def save(self, path:str):
        torch.save(self.model.state_dict(), path)
    # Load the model weights.
    def load(self, path:str):
        # Load weights from path.
        weights = torch.load(path)
        # Set model weights.
        self.model.load_state_dict(weights)
    # Aggregate other model weights with this one.
    def aggregate(self, other_weights:OrderedDict[str, torch.Tensor]):
        # Got some help from: https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029.
        # Add this model's weights to list of weights.
        other_weights.append(self.model.state_dict())
        # Calculate mean per each key of the state dictionary.
        these_weights = self.model.state_dict()
        for k in these_weights:
            # Compute the mean by creating a stack from input tensors and taking the mean.
            these_weights[k] = torch.stack([state_dict[k] for state_dict in other_weights]).mean(0)
        # Set this model's weights to the mean of all weights.
        self.model.load_state_dict(these_weights)
        # Return weights if desired.
        return these_weights

# Dataset class.
class torch_MNIST:
    # Custom dataset class to use with PyTorch.
    class torch_dataset(Dataset):
        def __init__(self, data):
            self.features = data[0]
            self.targets = data[1]
        def __len__(self):
            return len(self.features)
        def __getitem__(self, i):
            return self.features[i], self.targets[i]
    # Initialization.
    def __init__(self, filename:str, split:float=0.8, log:simlog.Log=None):
        # Set display output to either log or print.
        if log is not None:
            self.display = log.log
        else:
            self.display = print
        self.display(f"Loading file from: {filename}")
        # File contains pickled list of tuples (feature vector, label).
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set the point to split the data.
        split_point = int(split * len(data))
        # Initialize arrays.
        train_X, train_y, test_X, test_y = [], [], [], []
        # Go through each data point and add to train or test.
        for i, datum in enumerate(data):
            if i < split_point:
                train_X.append(datum[0])
                train_y.append(datum[1])
            else:
                test_X.append(datum[0])
                test_y.append(datum[1])
        # Convert arrays to tensors.
        train_X = torch.stack(train_X, dim=1)[0]
        train_y = torch.Tensor(train_y).type(torch.LongTensor)
        test_X = torch.stack(test_X, dim=1)[0]
        test_y = torch.Tensor(test_y).type(torch.LongTensor)
        # Create datasets.
        self.train_dataset = self.torch_dataset((train_X, train_y))
        self.test_dataset = self.torch_dataset((test_X, test_y))
        # Create dataloaders.
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)

# SELECTOR MODEL AND DATA - METHODS SHOULD CALL PYTORCH OR TENSORFLOW MODEL/DATA CLASSES.
class Model(torch_Model):
    def __init__(self, path=None, log=None):
        super().__init__(path, log=log)
class Data(torch_MNIST):
    def __init__(self, path=None, log=None):
        super().__init__(path, log=log)

# TENSORFLOW MODEL AND DATA - WARNING: AGGREGATION METHOD NOT COMPLETE.

# class tf_Model():
#     # Initialization function.
#     def __init__(self, model_generator, load_file=None):
#         # Define the model here. It's only called when a prior model does not exist.
#         if load_file is None:
#             # Call the pass thru model generator function.
#             self.model = model_generator()
#         # If a load file has been given, load model from the file.
#         else:
#             self.load(load_file)
#         # Define the data here.
#         pass
#     # Carries out training.
#     def train(self, data):
#         history = self.model.fit(data[0], data[1], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=False)
#     # Method to save model.
#     def save(self, filename):
#         self.model.save(filename)
#     # Method to load model.
#     def load(self, filename):
#         self.model = tf.keras.models.load_model(filename)

# class tf_Data():
#     # Initialization function takes and parses data into training and test.
#     def __init__(self):
#         # Store data from generator function. <--- This is where to change your data.
#         self.data = TweetData(DATA_FILE, ENCODER)
#         # Split the data.
#         self.split()
#     # Function to split data.
#     def split(self):
#         # Cut up the indices.
#         total = len(self.data[1])
#         fractions = (sum(TRAIN_TEST_VAL_SPLIT[:1]), sum(TRAIN_TEST_VAL_SPLIT[:2]), sum(TRAIN_TEST_VAL_SPLIT))
#         if fractions[-1] > 1:
#             raise IndexError("Check TRAIN_TEST_VAL_SPLIT and ensure proper ratios.")
#         splits = (np.array(fractions) * total)
#         splits = [int(x) for x in splits]
#         print("Splits:", splits)
#         # Store the dataset splits.
#         self.training_data = (self.data[0][:splits[0]], self.data[1][:splits[0]])
#         self.validation_data = (self.data[0][splits[0]:splits[1]], self.data[0][splits[0]:splits[1]])
#         self.testing_data = (self.data[0][splits[1]:splits[2]], self.data[0][splits[1]:splits[2]])
#     pass

# Main function to test.
if __name__ == "__main__":

    # TENSORFLOW TEST CODE.
    # data = tf_Data()
    # model1 = tf_Model(tf_LSTM)
    # model1.train(data.training_data)
    # model1.save("model.h5")
    # model2 = tf_Model()
    # model2.load("model.h5")
    # model2.train(data.training_data)
    # os.remove("model.h5")

    # PYTORCH TEST CODE.
    print("\nTESTING DATA AND MODEL...\n")
    data = Data("./data/test_client.data")
    model = Model()
    # Train the model.
    for i in range(10):
        print(f"Model 1 {i+1}\n-------------------------------")
        model.train(data.train_dataloader)
        model.test(data.test_dataloader)
    # Save the model.
    print("\nTESTING SAVING AND LOADING...\n")
    model.save("test_model.torch")
    # Load a new model.
    model_reborn = Model("test_model.torch")
    # Train the new model (should pick up where we left off).
    for i in range(10):
        print(f"Model 2 {i+1}\n-------------------------------")
        model_reborn.train(data.train_dataloader)
        model_reborn.test(data.test_dataloader)
    # Test the first and second models.
    print("\nTESTING AGGREGATION...\n")
    print("Model 1")
    print(model.model.state_dict())
    model.test(data.test_dataloader)
    print("Model 2")
    print(model_reborn.model.state_dict())
    model_reborn.test(data.test_dataloader)
    # Update the first model with aggregation with second model's weights.
    model.aggregate([model_reborn.model.state_dict(),])
    print(model.model.state_dict())
    # Test the model.
    model.test(data.test_dataloader)