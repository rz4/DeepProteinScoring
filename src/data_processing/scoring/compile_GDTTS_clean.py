'''
compile_GDTTS.py
Updated: 3/19/18

This script reads all the LGA output files stored in the /scores/ subdirectory
found in the data folder, parses out the GDTTS score and writes the scores and ids
to a csv file.

'''
import os
import numpy as np
from tqdm import tqdm

# Data Parameters
data_folder = '../../../data/T0898D1/'

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read scores
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'w') as fw:
        fw.write('ID,GDT_TS\n')
        for data_path in tqdm(sorted(os.listdir(data_folder+'scores'))):
            with open(data_folder+'scores/'+data_path, 'r') as f:
                mol_1 = []
                mol_2 = []
                for l in f:
                    if l.startswith('LGA'):
                        ll = l.split()
                        mol_1.append(ll[1])
                        mol_2.append(ll[3])
                    if l.startswith('SUMMARY(GDT)'):
                        score = float(l.split()[6])/100.0

                if mol_1 == mol_2:
                    fw.write(data_path + ',' + str(score) + '\n')
