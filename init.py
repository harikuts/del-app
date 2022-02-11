"""
Initialization Process: Creates directory structure and runs any 
other specific initialization processes.
"""

import os

NODELIST_FN = "nodelist.txt"

def main():
    print("Initializing...")
    # Get current directory.
    cur_dir = os.getcwd()
    # Create outbox.
    outbox_path = os.path.join(cur_dir, "outbox")
    print(f"Setting {outbox_path}...", end=' ')
    os.mkdir(outbox_path)
    print("Complete!")
    # Create inbox.
    inbox_path = os.path.join(cur_dir, "inbox")
    print(f"Setting {inbox_path}...", end=' ')
    os.mkdir(inbox_path)
    print("Complete!")
    # Get list of nodes.
    with open(NODELIST_FN, 'r') as f:
        nodelist = f.read().strip().split()
    # Create inbox subdirectories.
    for node in nodelist:
        # node = '_'.join(node.split('.'))
        node_path = os.path.join(inbox_path, node)
        print(f"Setting {node_path}...", end=' ')
        os.mkdir(node_path)
        print("Complete!")
    print("Initialization complete!")


if __name__ == "__main__":
    main()