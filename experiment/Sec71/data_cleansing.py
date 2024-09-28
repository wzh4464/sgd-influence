###
# File: /data_cleansing.py
# Created Date: Friday, September 27th 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 28th September 2024 1:27:13 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import argparse
import os
import numpy as np
import torch
import pandas as pd
from train import train_and_save
from infl import infl_true, infl_segment_true, infl_icml, infl_sgd, infl_lie
from DataModule import fetch_data_module
from NetworkModule import get_network
from logging_utils import setup_logging
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Function to load precomputed influence values based on influence type
def load_influence(infl_type, save_dir, seed):
    # infl_file name changes depending on the influence type (true, icml, sgd, etc.)
    infl_file = os.path.join(save_dir, f"infl_{infl_type}{seed:03d}.dat")
    return torch.load(infl_file)


# Function to find the most influential samples based on the top check_percentage%
def find_influential_samples(influence, check_percentage):
    n_samples = len(influence)
    # Calculate number of samples to check based on percentage
    n_check = int(n_samples * check_percentage / 100)
    # Sort indices based on influence values (smallest to largest)
    sorted_indices = np.argsort(influence)
    return sorted_indices[:n_check]


# Function to either load a pre-trained model or train a new onedef load_or_train_model(args, save_dir, logger):
def load_or_train_model(args, save_dir, logger):
    model_path = os.path.join(save_dir, f"sgd{args.seed:03d}.dat")
    if args.skip_initial_train and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        return torch.load(model_path, map_location=f"cuda:{args.gpu}")
    else:
        logger.info("Training new model")
        return train_and_save(
            args.target,
            args.model,
            args.seed,
            args.gpu,
            custom_n_tr=None,  # Use default if not specified
            custom_n_val=None,  # Use default if not specified
            custom_n_test=None,  # Use default if not specified
            custom_num_epoch=None,  # Use default if not specified
            custom_batch_size=None,  # Use default if not specified
            custom_lr=None,  # Use default if not specified
            compute_counterfactual=False,  # Compute leave-one-out (counterfactual) models
            save_dir=args.save_dir,
            logger=logger,
            relabel_csv=None,  # Initially no relabel CSV
        )




# Main function to run the data cleansing process
def main():
    args = parse_arguments()  # Parse command-line arguments
    save_dir = (
        args.save_dir or f"{args.target}_{args.model}"
    )  # Directory to save models/results
    full_save_dir = os.path.join(
        SCRIPT_DIR, save_dir
    )  # Full path to the save directory
    os.makedirs(full_save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(
        f"data_cleansing_{args.target}_{args.model}",
        args.seed,
        full_save_dir,
        level=log_level,
    )
    logger.info(f"Starting experiment with args: {args}")

    # Step 1: Load or train the initial model
    train_result = load_or_train_model(args, full_save_dir, logger)
    logger.info("Initial training completed or model loaded")

    # Step 2: Load influence data
    try:
        influence = load_influence(args.type, full_save_dir, args.seed)
        logger.info(f"Loaded influence for type: {args.type}")
    except FileNotFoundError:
        logger.error(f"Influence file not found for type: {args.type}")
        return

    # Step 3: Find the top influential samples based on check_percentage
    check_indices = find_influential_samples(influence, args.check)
    logger.info(f"Found {len(check_indices)} influential samples for {args.type}")

    # Step 4: Load existing relabeled indices
    relabeled_indices_file = os.path.join(
        full_save_dir, f"relabeled_indices_{args.seed:03d}.csv"
    )
    relabeled_indices_df = pd.read_csv(relabeled_indices_file)
    relabeled_indices = relabeled_indices_df["relabeled_indices"].values

    # Step 5: Remove influential samples from the relabeled set
    corrected_indices = np.intersect1d(
        check_indices, relabeled_indices
    )  # Find samples to be "corrected"
    updated_relabeled_indices = np.setdiff1d(
        relabeled_indices, corrected_indices
    )  # Update relabel indices
    num_corrected = len(relabeled_indices) - len(
        updated_relabeled_indices
    )  # Number of corrections made

    logger.info(f"Corrected {num_corrected} labels")

    # Step 6: Save the new relabeled indices to a CSV file
    # Modify filename to include check_{check}
    updated_relabeled_indices_file = os.path.join(
        full_save_dir, f"relabeled_indices_{args.type}_check_{args.check}_{args.seed:03d}.csv"
    )
    updated_relabeled_indices_df = pd.DataFrame(
        {"relabeled_indices": updated_relabeled_indices}
    )
    updated_relabeled_indices_df.to_csv(updated_relabeled_indices_file, index=False)
    logger.info(f"Updated relabeled indices saved to {updated_relabeled_indices_file}")

    # Step 7: Retrain the model with updated relabel CSV
    logger.info("Retraining model with updated relabel indices")
    final_result = train_and_save(
        args.target,
        args.model,
        args.seed,
        args.gpu,
        custom_n_tr=train_result["n_tr"],
        custom_n_val=train_result["n_val"],
        custom_n_test=train_result["n_test"],
        compute_counterfactual=False,  # Disable counterfactual model computation for retraining
        save_dir=f"{save_dir}",
        logger=logger,
        relabel_csv=updated_relabeled_indices_file,  # Use updated relabel CSV file
    )

    # Step 8: Output final results to a TXT file
    check_number = len(check_indices)
    fix_number = num_corrected
    left_unfix_number = len(updated_relabeled_indices)
    final_accuracy = final_result['test_accuracies'][-1]

    # Write these details to a .txt file
    output_txt_file = os.path.join(full_save_dir, f"summary_{args.type}_check_{args.check}_{args.seed:03d}.txt")
    with open(output_txt_file, "w") as f:
        f.write(f"Check number: {check_number}\n")
        f.write(f"Fix number: {fix_number}\n")
        f.write(f"Left unfix number: {left_unfix_number}\n")
        f.write(f"Final accuracy: {final_accuracy}\n")

    logger.info(f"Final test accuracy: {final_accuracy}")
    logger.info(f"Final training loss: {final_result['train_losses'][-1]}")
    logger.info(f"Final validation loss: {final_result['main_losses'][-1]}")



# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run experiment with data relabeling and influence calculation"
    )
    parser.add_argument("--target", required=True, type=str, help="Target dataset")
    parser.add_argument("--model", required=True, type=str, help="Model type")
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument("--gpu", required=True, type=int, help="GPU index")
    parser.add_argument(
        "--relabel", type=float, default=10, help="Percentage of data to relabel"
    )
    parser.add_argument(
        "--check",
        type=float,
        required=True,
        help="Percentage of data to check based on influence",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="true",
        choices=["true", "segment_true", "icml", "sgd", "lie"],
        help="Type of influence to calculate",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--skip_initial_train",
        action="store_true",
        help="Skip initial training and use existing model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
