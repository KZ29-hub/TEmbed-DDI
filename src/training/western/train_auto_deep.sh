#!/bin/bash

# # Define your paths and parameters
# TRAIN_PATH="/root/autodl-tmp/project-1101/data/western_data/chch_ddi_fold0/train_balanced_fold0.jsonl"
# TEST_PATH="/root/autodl-tmp/project-1101/data/western_data/chch_ddi_fold0/test_balanced.jsonl"
# CSV_PATH="/root/autodl-tmp/project-1101/models/res/chch_1101.csv"
# EPOCH=15
# BATCH_SIZE=16
# LR=5e-5
# DATASET_NAME="chch_nonor_cnn_bilstm_full"
# # Run the Python script with arguments
# python train_model_batch_cnn.py --train_path "$TRAIN_PATH" --test_path "$TEST_PATH" --epoch "$EPOCH" --batch_size "$BATCH_SIZE" --lr "$LR" --csv_file "$CSV_PATH" --dataset_name "$DATASET_NAME"


# Define your paths and parameters
TRAIN_PATH="/root/autodl-tmp/project-1101/data/western_data/deep_ddi_fold0/train_filtered_balanced_fold0.jsonl"
TEST_PATH="/root/autodl-tmp/project-1101/data/western_data/deep_ddi_fold0/test_filtered_balanced.jsonl"
CSV_PATH="/root/autodl-tmp/project-1101/models/res/deep_1101.csv"
EPOCH=20
BATCH_SIZE=16
LR=5e-5
DATASET_NAME="deep_nonor_cnn_bilstm_full"
# Run the Python script with arguments
python train_model_batch_cnn.py --train_path "$TRAIN_PATH" --test_path "$TEST_PATH" --epoch "$EPOCH" --batch_size "$BATCH_SIZE" --lr "$LR" --csv_file "$CSV_PATH" --dataset_name "$DATASET_NAME"
