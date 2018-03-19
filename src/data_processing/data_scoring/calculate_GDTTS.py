'''
calculate_GDTTS.py
Updated: 3/19/18

'''
import os
import numpy as np
from mpi4py import MPI

# Data Parameters
data_folder = '../../../data/T0868-D1/'
lga_command = '-3 -ie -o0 -sda -d:4'
seed = 64579

################################################################################

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

        if not os.path.exists(data_folder+'scores'): os.mkdir(data_folder+'scores')

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

    # Target path and id
    target_id = data_folder.split('/')[-2]
    target_path = data_folder + target_id + '.pdb'

    # Read Target PDB
    with open(target_path, 'r') as f: target_data = f.readlines()

    for i in range(len(tasks)):

        # Task parameters
        pdb_path = tasks[i]
        pdb_id = pdb_path.split('/')[-1][:-4]

        # Read PDB
        with open(pdb_path, 'r') as f: pdb_data = f.readlines()

        # Write combined PDB data
        with open('LGA/MOL2/'+pdb_id+'.'+target_id, 'w') as f:
            f.write('MOLECULE ' + pdb_id + '\n')
            f.writelines(pdb_data)
            if "END" not in pdb_data[-1]: f.write('END\n')
            f.write('MOLECULE ' + target_id + '\n')
            f.writelines(target_data)
            if "END" not in target_data[-1]: f.write('END\n')

        # Calculate GDT_TS using LGA
        lga_syscall = "LGA/lga.linux " + lga_command + ' ' + pdb_id+'.'+target_id
        os.system(lga_syscall + ' >> ' + data_folder + 'scores/' + pdb_id)
        os.remove('LGA/MOL2/'+pdb_id+'.'+target_id)
