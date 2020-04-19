#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=1GB
##SBATCH --gres=gpu:1
#SBATCH --job-name=test_email_from_noah
#SBATCH --mail-type=END
#SBATCH --mail-user=fg476@nyu.edu
#SBATCH --output=slurm_%j.out


python hello.py
