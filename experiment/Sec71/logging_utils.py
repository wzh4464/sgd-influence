###
# File: /logging_utils.py
# Created Date: Monday, September 16th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 16th September 2024 9:38:35 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
from datetime import datetime
import os


def setup_logging(filename, seed, output_dir="logs", level=logging.DEBUG):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%m%d%H%M%S")

    # Create the log filename
    log_filename = f"{filename}_{seed}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)

    # Configure the root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_path,
        filemode="a",  # Append mode
    )

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger("").addHandler(console_handler)

    logger = logging.getLogger("")
    logger.info(f"Logging setup completed. Log file: {log_path}")

    return logger
