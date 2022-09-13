import os

# RX/TX services.
PORT = 1245
DATASIZE = 1024

# Training and aggregation.
NUM_EPOCHS = 10
AGGREGATION_QUOTA = 2

# List of node addresses.
NODELIST_FN = "nodelist.txt"

# General filenames.
MODEL_FN = "model.torch"
DATA_FN = "client.data"
# Path to access data.
DATA_PATH = os.path.join(os.getcwd(), "data", "fashion-mnist.csv")
# Path to access your own model.
MODEL_PATH = os.path.join(os.getcwd(), "model.torch")
# Paths to publish your model updates.
OUTBOX_PATH = os.path.join(os.getcwd(), "outbox")
PUBLISHED_MODEL_PATH = os.path.join(OUTBOX_PATH, "model.torch")
# Paths to receive models.
INBOX_PATH = os.path.join(os.getcwd(), "inbox")
# Aggregated model filename and path to store aggregated model.
AGG_MODEL_FN = "model.agg.torch"
AGG_MODEL_PATH = os.path.join(os.getcwd(), AGG_MODEL_FN)