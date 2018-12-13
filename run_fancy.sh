#!/bin/bash

#SBATCH --time=16:00:00   # walltime
#SBATCH --ntasks=14   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH --gres=gpu:0
#SBATCH --output="/fslhome/tarch/compute/312/run_fancy3.slurm"
#SBATCH --constraint rhel7

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

cat /etc/os-release
cat /etc/redhat-release

module purge
source /fslhome/tarch/.bashrc

# module load miniconda/
export PATH="/fslhome/tarch/anaconda3/bin:$PATH"
export PATH="/fslhome/tarch/anaconda3/envs/main/bin:$PATH"
#conda activate main
#source activate main
#source activate /fslhome/tarch/anaconda3/envs/main

which python

#/zhome/tarch/compute/handwriting
cd /fslhome/tarch/compute/312/hopfield
python RUN.py
#--continue_train

# To run:
#sbatch ./run.sh
#sbatch /fslhome/tarch/compute/handwriting/run.sh 
#squeue -u tarch


#pip install git+https://github.com/Tahlor/utils.git
