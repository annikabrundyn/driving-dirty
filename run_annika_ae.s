#!/bin/bash -l

#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=50000
#SBATCH --time=05:00:00
#SBATCH --job-name=ab8690
#SBATCH --mail-user=ab8690@nyu.edu
#SBATCH --output=slurm_%j.out

# activate conda env
. ~/.bashrc
module load anaconda3/5.3.1
conda activate dirty-driving
conda install -n dirty-driving nb_conda_kernels

# -------------------------
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------

# run script from above
python src/autoencoder/autoencoder.py --link '/scratch/ab8690/DLSP20Dataset/data' --gpus 1 --batch_size 14 --precision 16 --max_epochs 100

