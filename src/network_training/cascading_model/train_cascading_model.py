'''
train_cascading_model.py
Updated: 3/15/18

This script is used to train a cascading-CNN which is a series of binary classifiers
which determine whether a given input feature grid of protein structures are greater
than or less than a specified GDT score threshold. This cascading-CNN approach has
shown greater capacity at binning input data into correct GDT ranges than a
multi-class network.

'''
import sys; sys.path.insert(0, '../')
import os
import h5py as hp
import numpy as np
from tqdm import tqdm
from networks import *
from keras.utils import to_categorical as one_hot
from sklearn.model_selection import train_test_split

# Network Training Parameters
epochs = 1
batch_size = 100
model_def = PairwiseNet_v2
model_folder = '../../../models/T0882_casc_pairwise/'

# Data Parameters
data_path = '../../../data/T0882/pairwise_data.hdf5'
ranks = [0.5, 0.6, 0.7, 0.8, 0.9]

################################################################################

split = [0.7, 0.1, 0.2]
seed = 678452

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(model_folder): os.mkdir(model_folder)

    # Gather ids and GDT-TS scores from csv
    ids = []
    scores = []
    data_folder = '/'.join(data_path.split('/')[:-1]) +'/'
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            x = lines[i].split(',')
            if i >= 1:
                id_ = x[0]; score = float(x[1])
                ids.append(id_); scores.append(score)
    x_data = np.array(ids)
    scores = np.array(scores)

    # Split training and test data
    x_data, x_test, y_scores, y_test_scores = train_test_split(x_data, scores, test_size=split[2], random_state=seed)

    # Load HDF5 dataset
    f = hp.File(data_path, "r")
    data_set = f['dataset']

    # Train Rankings
    for i in range(len(ranks)):

        rank = ranks[i]

        # Load Model
        if i == 0:
            model, loss, optimizer, metrics = model_def(2)
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model.summary()
        else: model.load_weights(model_path+'.hdf5')
        model_path = model_folder+model_def.__name__+'_'+str(rank)

        # Split into binary classification problem.
        print('Forming Binary Classification At Threshold:', rank)
        y_data = []
        for score in y_scores:
            if score >= rank: y_data.append(1)
            else: y_data.append(0)
        y_data = np.array(y_data)
        print('Positive:',len(np.where(y_data == 1)[0]),'Negative:', len(np.where(y_data == 0)[0]))

        # Split file paths into training, test and validation
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=split[1]/(split[0]+split[1]), random_state=seed)

        # Training Loop
        history = []
        best_val_loss = None
        for epoch in range(epochs):
            print("Epoch", epoch, ':', "Score Threshold:", rank)

            # Fit training data
            print('Fitting:')
            train_status = []
            batch_x = []
            batch_y = []
            for j in tqdm(range(len(x_train))):
                try: x = np.array(data_set[x_train[j]])
                except: continue
                batch_x.append(x)
                y = one_hot(y_train[j], num_classes=2)
                batch_y.append(y)
                if len(batch_x) == batch_size or j+1 == len(x_train):
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    output = model.train_on_batch(batch_x, batch_y)
                    batch_x = []
                    batch_y = []
                    train_status.append(output)

            # Calculate training loss and accuracy
            train_status = np.array(train_status)
            train_loss = np.average(train_status[:,0])
            train_acc = np.average(train_status[:,1])
            print('Train Loss ->', train_loss)
            print('Train Accuracy ->', train_acc,'\n')

            # Test on validation data
            print('Evaluating:')
            val_status = []
            batch_x = []
            batch_y = []
            for j in tqdm(range(len(x_val))):
                try: x = np.array(data_set[x_val[j]])
                except: continue
                batch_x.append(x)
                y = one_hot(y_val[j], num_classes=2)
                batch_y.append(y)
                if len(batch_x) == batch_size or j+1 == len(x_val):
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    output = model.test_on_batch(batch_x, batch_y)
                    batch_x = []
                    batch_y = []
                    val_status.append(output)

            # Calculate validation loss and accuracy
            val_status = np.array(val_status)
            val_loss = np.average(val_status[:,0])
            val_acc = np.average(val_status[:,1])
            print('Val Loss ->', val_loss)
            print('Val Accuracy ->', val_acc,'\n')

            if best_val_loss == None or val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save weights of model
                model.save_weights(model_path+'.hdf5')

            history.append([rank, epoch, train_loss, train_acc, val_loss, val_acc])

    # Get ranks of test set and store in dict
    ranks_list = []
    for i in range(len(x_test)):
        x = x_test[i]
        y = y_test_scores[i]
        rank = 0
        rankss = ranks + [1.0]
        for j in range(len(rankss)):
            if rankss[j] - y > 0:
                rank = j; break
        ranks_list.append([rank, 0])
    rankings_dict = dict(zip(x_test, ranks_list))

    # Evaluate test data
    possible_hits = 0
    for i in range(len(ranks)):
        rank = ranks[i]

        # Load weights of best model
        model_path = model_folder+model_def.__name__+'_'+str(rank)
        model.load_weights(model_path+'.hdf5')

        # Get inference results
        print("Running Inference On Threshold:", rank)
        batch_x = []
        batch_j = []
        for j in tqdm(range(len(x_test))):
            if rankings_dict[x_test[j]][1] == i:
                try: x = np.array(data_set[x_test[j]])
                except: continue
                possible_hits += 1
                batch_x.append(x)
                batch_j.append(j)
            if len(batch_x) == batch_size or j+1 == len(x_test):
                batch_x  = np.array(batch_x)
                if len(batch_x) != 0:
                    ss = model.predict_on_batch(batch_x)
                    for k in range(len(ss)):
                        s = int(np.argmax(ss[k]))
                        if s == 1: rankings_dict[x_test[batch_j[k]]][1] += 1
                batch_x = []
                batch_j = []

    # Measure accuracy of rankings
    hits = 0
    for key in rankings_dict.keys():
        ranking = rankings_dict[key]
        if ranking[0] == ranking[1]:
            hits += 1
    test_acc = float(hits) / len(possible_hits)
    print("Test Accuracy:", test_acc)

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [acc]: ' + str(test_acc)
    np.savetxt(model_folder+'results.csv', history, fmt= '%1.3f', delimiter=', ',
               header='LABELS: threshold, epoch, loss, acc, val_loss, val_acc',
               footer=test_footer)
