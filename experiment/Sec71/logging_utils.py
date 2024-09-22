###
# File: /logging_utils.py
# Created Date: Monday, September 16th 2024
# Author: Zihan
# -----
# Last Modified: Sunday, 22nd September 2024 12:09:47 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
from datetime import datetime
import os

def setup_logging(name, seed, save_dir, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    os.makedirs(save_dir, exist_ok=True)
    f_handler = logging.FileHandler(os.path.join(save_dir, f"log_{seed}.log"), mode='w')
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

