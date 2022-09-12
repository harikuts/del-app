"""
Learning Library:
Classes and methods that pertain to the learning process.
"""
from hmac import compare_digest
import sys
import pickle
from typing import OrderedDict
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import  SubsetRandomSampler 
import argparse
# import tensorflow

import simlog

"""
Global Variables:
Define any global variables needed by your scenarios.
"""
# Variable for CommandLine Interface: 
RUN_MNIST = True

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

    class FashionCNN(nn.Module):
        def __init__(self):
            super(torch_Model.FashionCNN, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
    # Initialization function.
    def __init__(self, load_path:str=None, log:simlog.Log=None):
        # Set display output to either log or print.
        if log is not None:
            self.display = log.log
        else:
            self.display = print

        # Instantiate the model.
        self.model = self.FashionCNN()
        """
        if RUN_MNIST:
            self.model = self.SimpleModel_MNIST(28*28, 256, 10)
        else:
            self.model = self.FashionCNN()
        """

        if load_path is not None:
            # If load file is provided, load model from there.
            self.load(load_path)
        # Loss function and optimizer variables.
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    
    # Training function.
    def train(self, data:DataLoader):
        size = len(data.dataset)
        self.model.train()
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

    # Check to see that state dictionaries have the same shape
    def compare_state_dict(self, other_weights):
        # Checking to see that the models have the same number of layers 
        if(len(other_weights) != len(self.model.state_dict())):
            print("Models have different number of layers")
            return False

        #Checking to see that the model layers have the same number of weights
        num_weights_in_tensor = []
        for key in other_weights:
            num_weights_in_tensor.append(other_weights[key].shape)
        
        i = 0
        for key in self.model.state_dict():
            if(num_weights_in_tensor[i] != self.model.state_dict()[key].shape):
                print("Model layer has different number of weights")
                return False
            i = i + 1

        return True


    # Aggregate other model weights with this one.
    def aggregate(self, other_weights:OrderedDict[str, torch.Tensor]):
        # Got some help from: https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029.

        if(self.compare_state_dict(other_weights[0]) == False):
            sys.exit()

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

    def getModelWeights(self):
        return self.model.state_dict()

# Dataset class.
class torch_FashionMNIST:
    #Custom Dataset class:
    class torch_fashion():
        def __init__(self, data, transform = None):
            self.fashion_MNIST = list(data.values)
            self.transform = transform
            
            label = []
            image = []
            
            for i in self.fashion_MNIST:
                # first column is of labels.
                label.append(i[0])
                image.append(i[1:])
            self.labels = np.asarray(label)
            # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
            self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

        def __getitem__(self, index):
            label = self.labels[index]
            image = self.images[index]
            
            if self.transform is not None:
                image = self.transform(image)

            return image, label

        def __len__(self):
            return len(self.images)


    def __init__(self, filename:str, split:float=0.8, log:simlog.Log=None): 
        
        if log is not None:
            self.display = log.log
        else:
            self.display = print
        self.display(f"Loading file from: {filename}")

        train_csv = pd.read_csv(filename)
        test_csv = pd.read_csv("./data_repo/fashion-mnist_test.csv")

        self.train_dataset = self.torch_fashion(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
        self.test_dataset = self.torch_fashion(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100)


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
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=True)

# SELECTOR MODEL AND DATA - METHODS SHOULD CALL PYTORCH OR TENSORFLOW MODEL/DATA CLASSES.
class Model(torch_Model):
    def __init__(self, path=None, log=None):
        super().__init__(path, log=log)
class DataFashion(torch_FashionMNIST):
    def __init__(self, path=None, log=None):
            super().__init__(path, log=log)
class DataMNIST(torch_MNIST): 
    def __init(self, path=None, log=None):
        super().__init__(path, log=log)


# Main function to test.
if __name__ == "__main__":

    #Command Line interface code: 
    parser = argparse.ArgumentParser(description="Simulate Machine Learning in a Federated Learning setting")
    parser.add_argument('-m', '--MNIST', action='store_true', help = "Run the MNIST numbers dataset.")
    parser.add_argument('-f', '--fashionMNIST', action='store_true', help = "Run based on the FashionMNIST dataset.")
    args = parser.parse_args()

    if args.MNIST is True and args.fashionMNIST is True:
        print("Cannot run both MNIST and FashionMNIST")
        sys.exit()
    elif args.fashionMNIST is True: 
        RUN_MNIST = False
    elif args.MNIST is True: 
        RUN_MNIST = True
    else:
        print("Argument must be provided")
        sys.exit()

    if(RUN_MNIST):
        data1_path = "./data_repo/node1/client.data"
        data2_path = "./data_repo/node2/client.data"
        data1 = DataMNIST(data1_path)
        data2 = DataMNIST(data2_path)
    else:
        data1_path = "./data_repo/node1/fashion-mnist_train.csv"
        data2_path = "./data_repo/node2/fashion-mnist_train.csv"
        data1 = DataFashion(data1_path)
        data2 = DataFashion(data2_path)


    # PYTORCH TEST CODE.
    # Load data, train model, and save for Node 1.

    print("\nTESTING DATA AND MODEL FOR NODE 1...\n")
    model1 = Model()
    # Train the model.
    for i in range(1):
        print(f"Node 1 {i+1}\n--------------------------------")
        model1.train(data1.train_dataloader)
        model1.test(data1.test_dataloader)
    # Save the model.
    print("\nSAVING NODE 1 MODEL...\n")
    model1.save("test_model_1.torch")
    # Load data, train model, and save for Node 1, same process.
    print("\nTESTING DATA AND MODEL FOR NODE 2...\n")
    model2 = Model()
    # Train the model.
    for i in range(1):
        print(f"Node 2 {i+1}\n--------------------------------")
        model2.train(data2.train_dataloader)
        model2.test(data2.test_dataloader)
    # Save the model.
    print("\nSAVING NODE 2 MODEL...\n")
    model2.save("test_model_2.torch")

    # Load Node 1 model weights into new model.
    print(f"\nLOADING NODE 1 MODEL AND TRAINING...\n")
    model1_reloaded = Model("test_model_1.torch")
    # Train the new model (should pick up where we left off), don't save.
    for i in range(1):
        print(f"Node 1 Reloaded {i+1}\n--------------------------------")
        model1_reloaded.train(data1.train_dataloader)
        model1_reloaded.test(data1.test_dataloader)

    # Reload original models.
    print(f"\nRELOADING MODELS FOR AGGREGATION...\n")
    model1 = Model(path="test_model_1.torch")
    model2 = Model(path="test_model_2.torch")
    # Now, try aggregating the originals (model1 and model2), after testing them first.
    print(f"\nOriginal Node 1 Model Test\n")
    model1.test(data1.test_dataloader)
    print(f"\nOriginal Node 2 Model Test")
    model2.test(data2.test_dataloader)
    # breakpoint()
    # Aggregate the model using both model objects, and test both.
    print("\nAggregating Node 2 model into Node 1 model...\n")
    model1.aggregate([model2.model.state_dict(),])

    print("\nAggregating Node 1 model into Node 2 model...\n")
    model2.aggregate([model1.model.state_dict(),])
    # Test models again to see post-aggregation accuracy.
    print(f"\nOriginal Node 1 Model Test - Post-Aggregation\n")
    model1.test(data1.test_dataloader)
    print(f"\nOriginal Node 2 Model Test - Post-Aggregation")
    model2.test(data2.test_dataloader)
 
    # Now, we try training models post-aggregation.
    print(f"\nTesting Node 1 Model Training After Aggregation...\n")
    for i in range(1):
        print(f"Node 2 {i+1}\n--------------------------------")
        model2.train(data1.train_dataloader)
        model2.test(data1.test_dataloader)

    

    # # Test the first and second models.
    # print("\nTESTING LOADING AND CONTINUING TRAINING...\n")
    # print("Model 1")
    # print(model1.model.state_dict())
    # model1.test(data1.test_dataloader)
    # model
    
    # print("Model 2")
    # print(model_reborn.model.state_dict())
    # model_reborn.test(data.test_dataloader)
    # # Update the first model with aggregation with second model's weights.
    # model.aggregate([model_reborn.model.state_dict(),])
    # print(model.model.state_dict())
    # # Test the model.
    # model.test(data.test_dataloader)