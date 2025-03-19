#!/bin/bash

# Add vllm to PYTHONPATH
export PYTHONPATH="/home/username/Projects/DrivingVllmBench:$PYTHONPATH"

# Activate mamba environment
# mamba init && mamba activate driving_vllm_bench_minicpmv
export CUDA_VISIBLE_DEVICES=1
# Run the evaluation script with parameters
/home/username/mambaforge/envs/driving_vllm_bench_vllm_latest/bin/python /home/username/Projects/DrivingVllmBench/driving_vllm_arena/benchmarking/run_evaluation.py \
    --model llava_next_video_7b \
    --config /home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/model_configs/vllm/config_llava_next_video_7b.py \
    --question_path "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/question_bank/question_bank_1000.jsonl" \
    --batch_size 1 \
    --placement all \
    --output_dir "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses" 