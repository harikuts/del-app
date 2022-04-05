#!/bin/env bash

# Initialize DEL.
python3 init.py

# Start DEL running services.
python3 train.py &
python3 aggregate.py &
python3 tx.py &
python3 rx.py &

# Wait for any process to exit, get status.
wait -n
exit $?