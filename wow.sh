#!/bin/bash
#SBATCH --job-name wow
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:rtx

rm -r /ext/sanisoglum
mkdir /ext/sanisoglum

WORKDIR=/home/sanisoglum
cd "$WORKDIR" || exit 0  # Create and change to the specified directory

cp -a riffusion-model-v1 /ext/sanisoglum

ls /ext/sanisoglum

wget https://huggingface.co/riffusion/riffusion-model-v1/resolve/main/riffusion-model-v1.ckpt -O /ext/sanisoglum/riffusion-model-v1

echo Download complete

ls /ext/sanisoglum/riffusion-model-v1