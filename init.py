"""
Initialization Process: Creates directory structure and runs any 
other specific initialization processes.
"""

import os
from learn import Model

NODELIST_FN = "nodelist.txt"

def main():
    print("Initializing...")

    # Create all the necessary directories.
    print("Building directories...")
    # Get current directory.
    cur_dir = os.getcwd()
    # Create outbox.
    outbox_path = os.path.join(cur_dir, "outbox")
    print(f"\tSetting {outbox_path}...", end=' ')
    os.mkdir(outbox_path)
    print("Complete!")
    # Create inbox.
    inbox_path = os.path.join(cur_dir, "inbox")
    print(f"\tSetting {inbox_path}...", end=' ')
    os.mkdir(inbox_path)
    print("Complete!")
    # Get list of nodes.
    with open(NODELIST_FN, 'r') as f:
        nodelist = f.read().strip().split()
    # Create inbox subdirectories.
    for node in nodelist:
        # node = '_'.join(node.split('.'))
        node_path = os.path.join(inbox_path, node)
        print(f"\tSetting {node_path}...", end=' ')
        os.mkdir(node_path)
        print("Complete!")

    # Initialize the model.
    Model().save("model.h5")

    print("Initialization complete!")


if __name__ == "__main__":
    main()