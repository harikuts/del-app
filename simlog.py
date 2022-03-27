# Logging class for the different DEL processes.

import datetime

class Log:
    # Initialization functions should set the the target file and header.
    def __init__(self, header, log_file):
        self.header = header
        self.log_file = log_file
    
    # Log messages by writing to a file.
    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"({datetime.datetime.now()}) {self.header} :: {message}\n")