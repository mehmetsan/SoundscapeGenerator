#!/bin/bash
#SBATCH --job-name soundscape_generator
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:rtx

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd $WORKING_DIR

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
srun ~/miniconda3/bin/conda run \
    -n myenv \
    python -m ./preparation.py