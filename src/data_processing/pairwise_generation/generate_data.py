'''
generate_data.py
Updated: 3/29/18

This script is used to generate pairwise distance matricies used for
convolutional neural network training. The script will store representations
in npz files within a /pairwise_data/ subdirectory. This script is used specifically to
generate data used for CASP experiments.

'''
import os
import numpy as np
from mpi4py import MPI
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist
from itertools import combinations

# Data generation parameters
data_folder = '../../../data/T0/' # Path to data folder
pairwise_distance_bins = [i*5 for i in range(10)]

################################################################################

# Static Parameters
chain = 'A' # Chain Id might need to be changed for PDBs missing identifier
seed = 458762 # For random distribution of tasks using MPI
residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL']

def parse_pdb(path, chain):
    '''
    Method parses atomic coordinate data from PDB.

    Params:
        path - str; PDB file path
        chain - str; chain identifier

    Returns:
        data - np.array; PDB data

    '''
    # Parse residue, atom type and atomic coordinates
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        residue = None
        residue_data = []
        flag = False
        for row in lines:
            if row[:4] == 'ATOM' and row[21] == chain:
                flag = True
                if residue != row[17:20]:
                    data.append(residue_data)
                    residue_data = []
                    residue = row[17:20]
                atom_data = [row[17:20], row[12:16].strip(), row[30:38], row[38:46], row[47:54]]
                residue_data.append(atom_data)
            if row[:3] == 'TER' and flag: break
    data = np.array(data[1:])

    return data

def bin_pairwise_distances(protein_data, pairwise_distance_bins):
    '''
    Method bins pairwise distances of residue alpha carbons into 2D data grids.

    Params:
        protein_data - np.array;
        pairwise_distance_bins - list; list of bins used to bin pairwise distances

    Returns:
        binned_pairwise - np.array;

    '''
    # Get alpha carbons
    alpha_carbons = []
    for i in range(len(protein_data)):
        residue = np.array(protein_data[i])
        ac_i = np.where(residue[:,1] == 'CA')
        alpha_carbons.append(residue[ac_i][0])
    alpha_carbons = np.array(alpha_carbons)

    # Pairwise distances
    dist = np.array(pdist(alpha_carbons[:,2:]))
    labels = list(combinations(alpha_carbons[:,0],2))
    labels = np.array([i[0] + i[1] for i in labels])

    # Bin pairwise distances
    bin_x = []
    for r1 in residues:
        bin_y = []
        for r2 in residues:
            i = np.where(labels == r1+r2)
            H, bins = np.histogram(dist[i], bins=pairwise_distance_bins)
            H = gaussian_filter(H, 0.5)
            bin_y.append(H)
        bin_x.append(bin_y)
    binned_pairwise = np.array(bin_x)

    return binned_pairwise

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    # MPI task distribution
    if rank == 0:
        tasks = []

        if not os.path.exists(data_folder+'pairwise_data'): os.mkdir(data_folder+'pairwise_data')

        # Search for data directories
        for data_path in sorted(os.listdir(data_folder+'pdbs')):
            if data_path.endswith('.pdb'):
                tasks.append(data_folder+'pdbs/'+data_path)

        # Shuffle for random distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    for t in tasks:
        path = t
        if chain == None: chain == 'A'
        save_path = '/'.join(t.split('/')[:-2]) + '/pairwise_data/'+ t.split('/')[-1][:-3]+'npz'

        # Parse PDB
        protein_data = parse_pdb(path, chain)

        try:
            # Bin pairwise distances
            binned_pairwise_distances = bin_pairwise_distances(protein_data, pairwise_distance_bins)

            # Save data
            np.savez(save_path, binned_pairwise_distances)

            print("Generated:", '/'.join(save_path.split('/')[-3:]))

        except: print("Error generating data...")

    print("Data Generation Complete.")
