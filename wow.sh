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

ls /ext/sanisoglum/
ls /ext/sanisoglum/checkpoints
ls -ll /ext/sanisoglum/checkpoints/caches
