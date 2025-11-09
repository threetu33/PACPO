#!/bin/bash

# ----------------------------
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

NUM_PROCESSES=1
MAIN_PROCESS_PORT=20139

# ----------------------------

DATASET_CAT='Musical_Instruments'
DATASET_DIR='/home/dongyinpeng/ltw/PACPO/data/'$DATASET_CAT'_0_2022-10-2023-10'

accelerate launch --num_processes=$NUM_PROCESSES --config_file=accelerates/deepspeed_config.yaml \
                --main_process_port=$MAIN_PROCESS_PORT train.py \
                --model=qwen \
                --dataset_dir=$DATASET_DIR \
                --dataset_category=$DATASET_CAT \
                --train_batch_size=4 \
                --eval_batch_size=32 \
                --max_new_tokens=640 \
                --warmup_steps=32 \
                --seed=42 \
                --num_train_epochs=3 \
                --run_name='atry' \
                --group_size=4 \
                --trainer_relabel_topk=2 \
                --gen_temperature=2.0 \
                --gen_top_k=50

