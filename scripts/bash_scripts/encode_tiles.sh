#!/bin/bash

# Paramètres SLURM
#SBATCH --job-name=ctranspath
#SBATCH -c 5
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:P100:1
#SBATCH --mem=16GB

# Exécution du script Python
python encode_tiles.py \
    --apply_pca False\
    --device "cuda"\
    --gt_filepath "/cluster/CBIO/data1/ablondel1/WSI_vesper_data/Selected_Contour_annotation.xlsx" \
    --data_folder '/cluster/CBIO/data1/ablondel1/WSI_vesper_data/'\
    --embedding_folder '/cluster/CBIO/data1/ablondel1/WSI_vesper_data/Embeddings/Ctranspath/'\
    --save_tiles_img True


    # device = "mps"
    # gt_filepath = "/Users/aliceblondel/Desktop/WSI_vesper/data/Selected_Contour_annotation.xlsx"
    # data_folder = "/Users/aliceblondel/Desktop/WSI_vesper/data/"
    # embedding_folder = "/Users/aliceblondel/Desktop/WSI_vesper/data/Embeddings_complete/ctranspath/"