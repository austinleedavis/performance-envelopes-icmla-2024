#!/bin/bash
#SBATCH --job-name=Trainer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=32:00:00  # Adjust time requirement
#SBATCH --gres=gpu:1
#SBATCH --exclusive=user  # Ensure exclusive access to GRES at the user level
#SBATCH --constraint=h100

########################################
########################################

cat "$0" # output this file's contents to the log

module load anaconda/anaconda3
module load cuda/cuda-12.1
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd

#accelerate config default

curl -d "Training Started" https://ntfy.sh/awesomesauceisinteresting

# All args in the args file
python train_transformer/train-chess-gpt.py --config_dir train_transformer/configs/austindavis/chess-gpt2-uci-12x12x768

########################################
########################################