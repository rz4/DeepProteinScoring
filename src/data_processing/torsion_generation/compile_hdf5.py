'''
compile_hdf5.py
Updated: 3/29/18

This script is used to save all generated torsion data saved under
/torsion_data/ subdirectory from the defined data_folder into a singular
HDF5 file named torsion_data.hdf5.

'''
import os
import h5py as hp
import numpy as np

# Data folder path
data_folder = '../../../data/Test/'

################################################################################
seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Search for class folders withing dataset folder
    print("Searching data folders for entries...")
    x_data = []
    for data_path in sorted(os.listdir(data_folder+'torsion_data')):
        if data_path.endswith('.npz'):
            x_data.append(data_folder+'torsion_data/'+data_path)

    # Setup HDF5 file and dataset
    print("Storing to HDF5 file...")
    f = hp.File(data_folder+"torsion_data.hdf5", "w")
    grp = f.create_group("dataset")

    # Write training data
    for i in range(len(x_data)):
        torsion = np.load(x_data[i])['arr_0']
        dset = grp.create_dataset(x_data[i].split('/')[-1][:-4], torsion.shape, dtype='f')
        dset[:,:,:] = torsion[:,:,:]

    print("Compiling Complete:", data_folder+"torsion_data.hdf5")
