#!/bin/bash -x
#SBATCH --partition=t-gpu-gh200
#SBATCH --account=maelstrom
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=singlegpu_graceh.%j
#SBATCH --error=singlegpu_graceh_err.%j

module load slurm
#module load python/python-3.9.6 
#source /home/sashkboos/env/bin/activate
#nvidia-smi

srun docker run --gpus all --rm --ipc=host -e HOME=$HOME -e USER=$USER --mount type=bind,source=/data/,dst=/data/ --mount type=bind,source=/home/sashkboos/ens10-d3.7/baselines,dst=/work -w /home/sashkboos/ens10-d3.7/baselines nvcr.io/nvidia/pytorch:24.01-py3 python3 get_power_nvml.py
