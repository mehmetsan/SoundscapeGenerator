#!/bin/bash
#SBATCH --job-name transfer_folder
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user mehmet.sanisoglu@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 10G

scp -r /ext/sanisoglum/saved/models--riffusion--riffusion-model-v1 mehmetsanisoglu@192.168.1.4:/Users/mehmetsanisoglu/Desktop/files
