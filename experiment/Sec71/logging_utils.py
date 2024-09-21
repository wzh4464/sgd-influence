###
# File: /logging_utils.py
# Created Date: Monday, September 16th 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 21st September 2024 10:02:48 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
from datetime import datetime
import os


def setup_logging(filename, seed, output_dir=None, level=logging.DEBUG):
    # Create the logger
    logger = logging.getLogger(filename)

    # If the logger already has handlers, assume it's configured and return it
    if logger.handlers:
        return logger

    # Set the log level
    logger.setLevel(level)

    # Create the output directory if it doesn't exist
    if output_dir is None:
        output_dir = "logs"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%m%d%H%M%S")

    # Create the log filename
    log_filename = f"{filename}_{seed}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging setup completed. Log file: {log_path}")
    return logger
