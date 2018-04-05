'''
combine_hdf5.py
Updated: 3/27/18

The script will read the pairwise_data.hdf5 file from each listed data_folder
and combine them into a singular HDF5 file. This script assumes that the
pairwise_data.hdf5 file has already been generated for each data folder in the
data_folders list. The final combined HDF5 file will be saved under the path defined
in the combined_folder variable.

'''
import os
import h5py as hp
import numpy as np

# Data folder paths
data_folders = ['../../data/T0862D1/',
                '../../data/T0865D1/',
                '../../data/T0870D1/',
                '../../data/T0885D1/',
                '../../data/T0887D1/',
                '../../data/T0890D2/',
                '../../data/T0892D1/',
                '../../data/T0893D1/',
                '../../data/T0898D1/',
                '../../data/T0915D1/',
                '../../data/T0922D1/']

# Combined folder path
combined_folder = '../../../data/AlphaSet/'

################################################################################

seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(combined_folder): os.mkdir(combined_folder)

    '''
    # Combine CSVs
    print("Combining CSVs...")
    data = []
    for folder in data_folders:
        with open(folder+folder.split('/')[-2]+'.csv', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if i >= 1: data.append(folder.split('/')[-2]+'_'+lines[i])
    with open(combined_folder+combined_folder.split('/')[-2]+'.csv', 'w') as f:
        f.write("ID,GDT_TS\n")
        f.writelines(data)
    print('Complete:', combined_folder+combined_folder.split('/')[-2]+'.csv')
    '''

    # Combine HDF5 datasets
    print("Combining HDF5s...")
    c = hp.File(combined_folder+"pairwise_data.hdf5", "w")
    c_grp = c.create_group("dataset")
    for folder in data_folders:
        f = hp.File(folder+"pairwise_data.hdf5", "r")
        data_set = f['dataset']
        for key in list(data_set.keys()):
            x = np.array(data_set[key])
            dset = c_grp.create_dataset(folder.split('/')[-2]+'_'+key, x.shape, dtype='f')
            dset[:,:,:] = x[:,:,:]
    print("Complete:", combined_folder+"pairwise_data.hdf5")
