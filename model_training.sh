#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=25GB
##SBATCH --gres=gpu:1
#SBATCH --job-name=roadmap_baseline
#SBATCH --mail-type=END
##SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm_%j.out


python roadmap_baseline.py
