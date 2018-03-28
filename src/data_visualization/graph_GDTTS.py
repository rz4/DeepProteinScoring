'''
graph_GDTTS.py
Updated: 3/20/18

'''
import os
import numpy as np
import matplotlib.pyplot as plt

# Data parameters
data_folder = '../../../data/T0895D1/'

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load GDT_TS scores from csv
    scores = []
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'r') as f:
        lines = f.readlines()
        for l in lines[1:]:
            score = l.split(',')[1]
            scores.append(float(score))

    # Display histogram
    plt.hist(scores, 20, range=(0.0,1.0))
    plt.xlabel('GDT_TS')
    plt.ylabel('Number of Structures')
    plt.title('Histogram of GDT_TS')
    # plt.grid(True)
    plt.show()
