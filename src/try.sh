#!/bin/bash
#SBATCH --job-name=qwen_try
#SBATCH --output=/scratch/jl13122/nlp-project/src/log_qwen/outputs/output_%j.txt
#SBATCH --error=/scratch/jl13122/nlp-project/src/log_qwen/errors/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 5-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

source sloth/bin/activate
# Run your script
python finetune_qwen.py
