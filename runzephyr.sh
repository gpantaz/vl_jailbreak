#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=eval
#SBATCH --output=eval.%J.out
#SBATCH --gres gpu:1

.venv/bin/python run_zephyr.py
