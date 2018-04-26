# Directory /data/
Updated: 04/23/18

# Dataset Structure
This directory is used to store datasets. Each new dataset should be structured
as follows:

```
TargetD1/
|
|-TargetD1.pdb - Target crystal structure pbd file
|
|-/pdbs/ - Folder containing decoy pdb files

```

After generating data and scores, the final dataset will be structured as follows:

```
TargetD1/
|
|-TargetD1.pdb
|
|-/pdbs/
|
|-/pairwise_data/ - Numpy files of decoy pairwise data
|
|-pairwise_data.hdf5 - HDF5 of all decoy pairwise data
|
|-/scores/ - LGA output for all decoy structures
|
|-TargetD1.csv - CSV file of decoy ids and GDT-TS scores

```
