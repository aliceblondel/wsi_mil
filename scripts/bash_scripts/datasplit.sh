#!/bin/bash

# Paramètres SLURM
#SBATCH --job-name=ctranspath
#SBATCH -c 5
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:P100:1
#SBATCH --mem=16GB

# Exécution du script Python
python ./scripts.split_dataset.py\
    --table /path/master/table \
    --target_name output_variable \
    -k 4 \