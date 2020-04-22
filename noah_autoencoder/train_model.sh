#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:p40:1
#SBATCH --job-name=Train_Inception
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out


python noah_autoencoder.py
