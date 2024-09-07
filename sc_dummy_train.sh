#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks=1  # Keep ntasks=1, as torchrun will handle distribution
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:a40:2  # Request 2 GPUs

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu

source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory

# No need to set CUDA_VISIBLE_DEVICES, torchrun handles GPU allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export WANDB_DEBUG=true
export WANDB_HTTP_TIMEOUT=60  # Increase timeout
export MASTER_ADDR=$(hostname)  # Set the master address to the current node's hostname
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)        # Use a specific port for inter-process communication

# Use torchrun to launch the distributed training
torchrun --nproc_per_node=2 alternate_train.py  # Use 2 processes (one per GPU)
