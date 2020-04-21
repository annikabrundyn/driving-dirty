#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=Lightning_Autoencoder
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out


python grease_lightning.py
