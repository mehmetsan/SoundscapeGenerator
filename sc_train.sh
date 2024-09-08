#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1  # One task per node
#SBATCH --gpus-per-node=2  # Use 2 GPUs per node
#SBATCH --gres=gpu:a40:2  # Request 2 A40 GPUs explicitly
#SBATCH --partition allgroups
#SBATCH --mem 20G

source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO  # Enable NCCL debugging info
export NCCL_P2P_LEVEL=NVL  # Limit P2P communication to local GPUs
# export NCCL_SOCKET_IFNAME=eth0  # Uncomment and set your network interface if needed

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env train_model.py
