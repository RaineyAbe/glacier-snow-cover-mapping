#!/bin/bash
#SBATCH -J example              # job name
#SBATCH -o log_slurm.o%j        # output and error file name (%j expands to job$)
#SBATCH -n 28                   # total number of tasks requested
#SBATCH -N 1                    # number of nodes you want to run on
#SBATCH -p defq                 # queue (partition)
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 12.0 hours in this exam$

# source your .bashrc file (e.g., for PATH modifications and Conda initialization)
. ~/.bashrc

# activate conda environment
conda activate snow-cover-mapping

# run your code
python /home/raberle/scratch/snow-cover-mapping/scripts/snow_classification_pipeline.py
