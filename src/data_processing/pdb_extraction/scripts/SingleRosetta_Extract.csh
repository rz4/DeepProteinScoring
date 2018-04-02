#!/bin/bash -l
# ****************************************************************************
# Updated: 26 March 2018
# This script runs a single extraction on one node. Change the directories,
# the target ID, the number of cores, and the time to match the number of
# silent files and the target you are working with. Consulting
# "decoy_counts.txt" in /global/project/projectdirs/m1532/downloads/CASP12/
# will tell you how many PDB files must be extracted. Unfortunately, there is
# no reliable way to estimate the time it will take to extract a given number
# of PDB files, but we have found that some silent files containing 60K files 
# take roughly 9 hours, while one containing 90K takes roughly 12 hours. Silent
# files containing more than 130K PDBs will probably not be extractable within
# the 48 hour time limit on NERSC's Cori, although full extraction may not
# matter for your case. 
#
# This script assumes that the silent file has been placed in an
# appropriate subdirectory and has been unzipped. It will create a subdirectory 
# within the silent file directory called 'pdbs/' and will output all files there. 
#
# The quality of service (qos) operator can be set to regular or premium.
# Premium doubles the cost of the hours you allocate but provides rapid turn
# around, while regular will most likeluy result in a multi-day delay. 
# ****************************************************************************
#SBATCH -N 1
#SBATCH --qos=premium
#SBATCH -L SCRATCH
#SBATCH -t 04:00:00
#SBATCH -C haswell
# ****************************************************************************
cd /global/cscratch1/sd/tjosc/casp_data/t0943/0c1/
mkdir ./pdbs/
srun -N 1 -n 1 --cpu_bind=cores /global/project/projectdirs/m1532/RosettaCori/rosetta_src_2016.13.58602_bundle/main/source/bin/extract_pdbs.linuxgccrelease -in:file:silent_struct_type binary -in:file:silent *.out -out:prefix ./pdbs/ &
wait
