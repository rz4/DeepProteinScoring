'''
combine_hdf5.py
Updated: 3/9/18

'''
import os
import h5py as hp
import numpy as np

# Data folder paths
data_folders = ['../../../../data/T0882/',
                '../../../../data/T0866/',
                '../../../../data/T0867/',
                '../../../../data/T0868/']

# Combined folder path
combined_folder = '../../../../data/TargetSet0/'

################################################################################
seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Combine CSVs
    data = []
    for folder in data_folders:
        with open(folder+folder.split('/')[-2]+'.csv', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if i > 1: data.append(folder.split('/')[-2]+'_'+lines[i])
    with open(combined_folder+combined_folder.split('/')[-2]+'.csv', 'w') as f:
        f.write("ID,GDT_HA,GDT_MM,GDT_SC\n")
        f.writelines(data)

    # Combine HDF5 datasets
    c = hp.File(combined_folder+"torsion_pairwise_casp_data.hdf5", "w")
    c_grp = c.create_group("dataset")
    for folder in data_folders:
        f = hp.File(folder+"torsion_pairwise_casp_data.hdf5", "r")
        data_set = f['dataset']
        for key in list(data_set.keys()):
            x = np.array(data_set[key])
            dset = c_grp.create_dataset(folder.split('/')[-2]+'_'+key, x.shape, dtype='f')
            dset[:,:,:] = x[:,:,:]
