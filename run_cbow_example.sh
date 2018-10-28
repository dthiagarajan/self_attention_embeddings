#!/bin/bash
#SBATCH -J pbe_delta    # Job name
#SBATCH -o pbe_delta.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e pbe_delta.o%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 1   # Total number of CPU cores requrested
#SBATCH --mem=15000    # CPU Memory pool for all cores
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=default_gpu --gres=gpu:1 --nodelist=harpo

python3 cbow_example.py
