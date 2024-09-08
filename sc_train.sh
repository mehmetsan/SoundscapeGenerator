#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1  # Keep 1 task per node
#SBATCH --gres=gpu:a40:2  # Request 2 GPUs per node
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:a40:2

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu


source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Change to the specified directory

export CUDA_LAUNCH_BLOCKING=1

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env train_model.py