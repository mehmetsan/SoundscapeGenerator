#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:a40:1

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu

source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory


export CUDA_LAUNCH_BLOCKING=1

srun python train_model.py