#!/bin/bash
#SBATCH --job-name=RL_Lab5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=4          
#SBATCH --gres=gpu:1
#SBATCH --account=MST114564
#SBATCH --output=rl_training_%j.log 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vr

# 執行程式
python3 train2.py