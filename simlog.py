# Logging class for the different DEL processes.

import datetime
import logging


class Log:
    # Initialization functions should set the the target file and header.
    def __init__(self, header, log_file):
        self.header = header
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file)
        logging.getLogger().setLevel(logging.INFO)
    
    # Log messages by using the logging module.
    def log(self, message:str):
        lines = message.split('\n')
        for line in lines:
            logging.info(f"({datetime.datetime.now()}) {self.header} :: {line}")

    # Log messages by writing to a file.
    def old_log(self, message:str):
        lines = message.split('\n')
        with open(self.log_file, 'a') as f:
            for line in lines:
                f.write(f"({datetime.datetime.now()}) {self.header} :: {line}\n")