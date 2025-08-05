#!/bin/bash

# Define your paths and parameters
TRAIN_PATH="/root/autodl-tmp/multiview-project-1023/embeddings/tcm_data/tcm_cls2_tmp_n500/tcm_cls2_fold0/en/train_en.jsonl"
TEST_PATH="/root/autodl-tmp/multiview-project-1023/embeddings/tcm_data/tcm_cls2_tmp_n500/tcm_cls2_fold0/en/test_en.jsonl"
CSV_PATH="/root/autodl-tmp/multiview-project-1023/models/res/csv_res/tcm_1028.csv"
EPOCH=50
BATCH_SIZE=4
LR=1e-5
DATASET_NAME="tcm_en"
# Run the Python script with arguments
python train_model_batch_tcm.py --train_path "$TRAIN_PATH" --test_path "$TEST_PATH" --epoch "$EPOCH" --batch_size "$BATCH_SIZE" --lr "$LR" --csv_file "$CSV_PATH" --dataset_name "$DATASET_NAME"
