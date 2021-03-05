#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=derive_tasmax_tasmin
#SBATCH --account=isimip
#SBATCH --output=%x.log
#SBATCH --error=%x.log
#SBATCH --ntasks=1

/home/sitreu/.conda/envs/isi-cfact/bin/python -u create_tasmin_tasmax.py
