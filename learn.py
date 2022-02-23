"""
Learning Library:
Classes and methods that pertain to the learning process.
"""
import sys
import pickle
import numpy as np

# Import OS-specific tensorflow.
if sys.platform == 'darwin':
    print("OSX detected... ", end='')
    try:
        tf = __import__("tensorflow-macos")
        print("Imported Tensorflow (Metal) for MacOS.")
    except ImportError:
        import tensorflow as tf
        print("Could not import Tensorflow (Metal). Importing standard Tensorflow.")
else:
    import tensorflow as tf
    print("Windows or Linux detected... Importing standard Tensorflow.")

"""
Selector Classes:
These selector functions will point to the model and data you want to use.
You can create these functions in the model and data libraries below and
point those functions here.
"""
class Model():
    # Initialization function.
    def __init__(self):
        # Define the model here.
        self.model = LSTM()
        # Define the data here.
        pass

    # Carries out training.
    def train(self, data):
        pass

class Data():
    # Initialization function takes and parses data into training and test.
    def __init__(self):
        data = TweetData()
    pass

"""
Global Variables:
Define any global variables needed by 
"""
# Test Twitter-LSTM Scenario
SEQ_LEN = 2 # Number of preceeding words before next word.
DATA_FILE = "data/test_data.txt"
ENCODER_FILE = "data/encoder.pkl"
ENCODER = pickle.load(open(ENCODER_FILE, 'rb'))
VOCAB_SIZE = len(ENCODER) # Number of total words, which is mapped by the encoder.

"""
Model Library:
In this section, you can create models to use in your selector functions.
"""
def LSTM(vocab_size):
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
        print(prev_words, next_word)



    

# Main function to test.
if __name__ == "__main__":
    TweetData(DATA_FILE, ENCODER)