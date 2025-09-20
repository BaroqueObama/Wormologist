#!/bin/bash -l
##############################
#    Training Worm Model     #
##############################

#SBATCH --job-name="Ohio"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx5000:2
#SBATCH --time=8:00:00
#SBATCH --partition=p.hpcl8
#SBATCH --exclusive
#SBATCH --output=/fs/home/smola/code/CElegans/graph_matching/slurm_test_logs/slurm-530.out

source ~/.bashrc
conda activate worm
srun python -u /fs/home/smola/code/CElegans/graph_matching/process_worms.py --subgraph_size 530
