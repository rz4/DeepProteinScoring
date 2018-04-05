'''
graph_GDTTS.py
Updated: 3/20/18

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import *

# Data parameters
data_folders = ['../../data/T0862D1/',
                '../../data/T0865D1/',
                '../../data/T0870D1/',
                '../../data/T0885D1/',
                '../../data/T0887D1/',
                '../../data/T0890D2/',
                '../../data/T0893D1/',
                '../../data/T0892D1/',
                '../../data/T0922D1/']

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load GDT_TS scores from csv
    targets = []
    labels = []
    for data_folder in data_folders:
        scores = []
        labels.append(data_folder.split('/')[-2])
        with open(data_folder+data_folder.split('/')[-2]+'.csv', 'r') as f:
            lines = f.readlines()
            for l in lines[1:]:
                score = l.split(',')[1]
                scores.append(float(score))
        targets.append(scores)

    # Display histogram
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}
    cm = [Blues(float(i+1)/(len(targets)))[:3] for i in range(len(targets))]
    N, bins, patches = plt.hist(targets, 100, color=cm,range=(0.0,1.0), stacked=True, label=labels)
    plt.xlabel('GDT_TS', font)
    plt.ylabel('Number of Decoys', font)
    plt.title('Histogram of GDT_TS Scores', font)
    plt.legend()
    # plt.grid(True)
    plt.show()
