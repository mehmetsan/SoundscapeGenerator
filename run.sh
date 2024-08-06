#!/bin/bash
#SBATCH --job-name riffusion
#SBATCH --output log/out/%j.txt
#SBATCH --error log/err/%j.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 10:00
#SBATCH --partition allgroups
#SBATCH --mem 5G
#SBATCH --gres=gpu:rtx

srun ~/miniconda3/bin/conda run -n riffusion python testing.py