#!/bin/bash
#SBATCH -J pbe_delta    # Job name
#SBATCH -o pbe_delta.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e pbe_delta.o%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 1   # Total number of CPU cores requrested
#SBATCH --mem=100000    # CPU Memory pool for all cores
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=default_gpu --gres=gpu:1 --nodelist=nikola-compute03

if [ ! -d /scratch/datasets/models/ ]; then
        mkdir /scratch/datasets/models
fi

if [ ! -d /scratch/datasets/vocab.pkl ]; then
        cp -r /share/nikola/export/dt372/vocab.pkl /scratch/datasets/vocab.pkl
fi

if [ ! -d /scratch/datasets/training_data.pkl ]; then
        cp -r /share/nikola/export/dt372/training_data.pkl /scratch/datasets/training_data.pkl
fi

if [ ! -d /scratch/datasets/word_to_ix.pkl ]; then
        cp -r /share/nikola/export/dt372/word_to_ix.pkl /scratch/datasets/word_to_ix.pkl
fi

if [ ! -d /scratch/datasets/ix_to_word.pkl ]; then
        cp -r /share/nikola/export/dt372/ix_to_word.pkl /scratch/datasets/ix_to_word.pkl
fi

python3 cbow_example.py
