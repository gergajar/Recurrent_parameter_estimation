import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from os import listdir
from os.path import isfile, join

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)


data_path = os.path.join(PATH_TO_PROJECT, 'data')
#pickle.dump([train_set, val_set, test_set], open(os.path.join(data_path, 'data_set_dicts.pkl'), "wb"), protocol=2)
dataset = np.load(os.path.join(data_path, 'data_set_dicts.pkl'))

train = {}
val = {}
test = {}
sets = [train, val, test]

for set_idx in range(len(sets)):
    set_dict = dataset[set_idx]

    for k in set_dict[0].keys():
        sets[set_idx][k] = [d[k] for d in set_dict]

pickle.dump([train, val, test], open(os.path.join(data_path, 'data_set.pkl'), "wb"), protocol=2)