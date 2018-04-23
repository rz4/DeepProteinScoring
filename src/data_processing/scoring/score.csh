#!/bin/bash -l
#SBATCH -N 4
#SBATCH --qos=regular
#SBATCH -L SCRATCH
#SBATCH -t 01:00:00
#SBATCH -C haswell
# ****************************************************************************

module load python/3.6-anaconda-4.4
srun -n 128 python calculate_GDTTS.py
