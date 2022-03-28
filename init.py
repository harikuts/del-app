"""
Initialization Process: Creates directory structure and runs any 
other specific initialization processes.
"""

import os
from learn import Model
from simlog import Log

NODELIST_FN = "nodelist.txt"

def main():
    log = Log("INIT", os.path.join(os.getcwd(), "logs", "init.log"))
    log.log("Initializing...")

    # Create all the necessary directories.
    log.log("Building directories...")
    # Get current directory.
    cur_dir = os.getcwd()
    # Create outbox.
    outbox_path = os.path.join(cur_dir, "outbox")
    log.log(f"\tSetting {outbox_path}...")
    os.mkdir(outbox_path)
    log.log("Complete!")
    # Create inbox.
    inbox_path = os.path.join(cur_dir, "inbox")
    log.log(f"\tSetting {inbox_path}...")
    os.mkdir(inbox_path)
    log.log("Complete!")
    # Get list of nodes.
    with open(NODELIST_FN, 'r') as f:
        nodelist = f.read().strip().split()
    # Create inbox subdirectories.
    for node in nodelist:
        # node = '_'.join(node.split('.'))
        node_path = os.path.join(inbox_path, node)
        log.log(f"\tSetting {node_path}...")
        os.mkdir(node_path)
        log.log("Complete!")

    # Initialize the model.
    Model().save(os.path.join(os.getcwd(), "outbox", "model.torch"))

    log.log("Initialization complete!")


if __name__ == "__main__":
    main()