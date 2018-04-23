# Scoring Decoys

This directory contains all the scripts needed to calculate the GDT-TS scores of
a decoy dataset. The process of calculating the scores is as follow:

1. Create a new dataset folder in the [/data/](/data) directory, and copy your
decoy PDBs into a folder called 'pdbs' within your new dataset folder.
Copy the target crystal structure used for the comparison and make sure its
named the same as the newly created dataset folder plus the extension '.pdb'.
For more information on how this should look like, please refer to the README within the
[/data/](/data) directory.

2. Open [calculate_GDTTS.py](calculate_GDTTS.py) and
change the `data_folder` variable to the path to the dataset directory. The path should be
set relative to this file. This script utilizes the LGA package which is the standard
tool used to calculate GDT-TS. This process stores the output results
from the LGA package into a folder named '/scores/' within your dataset folder.
Scoring is parallelized with MPI. The [score.csh](score.csh) batch script
can be used to submit the job to Cori. Make sure to change number of nodes, number of cores,
and total time according to your dataset. This process takes one hour to run using 128
Haswell cores on Cori for a dataset of size 300,000.

> **IMPORTANT** : You may need to change the `lga_command` variable within the script
if the decoy structures do not align with target crystal structure. If they do not align,
remove the `-sda` flag from the command and insert the flags `-aa1:n1:n2` for the decoy
structure and `-aa2:n1:n2` for the target structure, replacing n1 and n2 with the residue
index range for each given structure.

3. Open [calculate_GDTTS.py](/src/data_processing/scoring/calculate_GDTTS.py) and
change the `data_folder` variable to the path to the dataset directory. This script will read
all the LGA output files generated from the last script and create a CSV file
containing the PDB ids and corresponding GDT-TS scores saved in the dataset directory.
