'''
compile_GDTTS.py
Updated: 3/19/18

'''
import os
import numpy as np
from tqdm import tqdm

# Data Parameters
data_folder = '../../../data/T0866/'

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read scores
    ids = []
    scores = []
    for data_path in tqdm(sorted(os.listdir(data_folder+'scores'))):
        with open(data_folder+'scores/'+data_path, 'r') as f:
            for l in f:
                if l.startswith('SUMMARY(GDT)'):
                    score = float(l.split()[6])/100.0
                    scores.append(score)
                    ids.append(data_path)

    # Write scores
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'w') as f:
        f.write('ID,GDT_TS\n')
        for i in range(len(ids)):
            f.write(ids[i] + ',' + str(scores[i]) + '\n')
