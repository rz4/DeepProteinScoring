'''
display_pairwise.py
Updated: 3/20/18

'''
import os
import h5py as hp
import numpy as np
import matplotlib.pyplot as plt

# Data parameters
data_folder = '../../data/T0882/'

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load pairwise data
    f = hp.File(data_folder+'pairwise_data.hdf5', "r")
    data_set = f['dataset']
    x = np.array(data_set[list(data_set.keys())[0]])

    # Display histogram
    for i in range(x.shape[2]):
        plt.imshow(x[:,:,i], cmap='Blues')
        plt.show()
