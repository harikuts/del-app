"""
Learning Library:
Classes and methods that pertain to the learning process.
"""
import sys

# Import OS-specific tensorflow
if sys.platform == 'darwin':
    try:
        tf = __import__("tensorflow-macos")
    except ImportError:
        import tensorflow as tf
else:
    import tensorflow as tf

"""
Selector Functions:
These selector functions will point to the model and data you want to use.
You can create these functions in the model and data libraries below and
point those functions here.
"""
class Model():
    pass

def data():
    pass


"""
Model Library:
In this section, you can create models to use in your selector functions.
"""
def LSTM(self, vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(1024, input_shape=(SEQ_LEN, vocab_size), return_sequences=True))
    model.add(tf.keras.layers.LSTM(1024))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    return model