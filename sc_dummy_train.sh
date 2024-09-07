#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4  # Ensure sufficient CPUs for each task
#SBATCH --gpus-per-task=1  # One GPU per task
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:a40:2

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu

source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory


export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

srun python alternate_train.py