#!/bin/bash
#SBATCH --job-name dummy_train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --nodes=1  # Force the job to run on a single node
#SBATCH --gres=gpu:a40:2  # Request 2 GPUs on the same node
#SBATCH --partition allgroups
#SBATCH --mem 20G

source /home/sanisoglum/miniconda3/bin/activate my_env

WORKDIR=/home/sanisoglum/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Change to the specified directory

export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL  # Limit to intra-node communication
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_SHM_DISABLE=0  # Enable shared memory communication
export NCCL_NET_GDR_LEVEL=0  # Disable GPUDirect RDMA (optional)

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env train_model.py
