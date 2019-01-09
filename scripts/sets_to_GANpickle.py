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

#kwargs["amp"], kwargs["freq"], kwargs["phase"]
def set_to_dicts_list(data, time_series_class, set):
    data = data[set]
    dicts_list = []
    for sequence_index in range(data['n_sequences']):
        sequence_dict = {'original_magnitud': data['real_values'][sequence_index],
                         'original_time': data['dense_time'][sequence_index],
                         'frequency': data['params'][sequence_index][1],
                         'amplitude': data['params'][sequence_index][0],
                         'phase': data['params'][sequence_index][2],
                         'class': time_series_class,}
        dicts_list.append(sequence_dict)
    return dicts_list

data_path = os.path.join(PATH_TO_PROJECT, 'data')
shapes = ["square", "sawtooth", "sinusoidal"]
train_set = []
val_set = []
test_set = []

for shape in shapes:
    print(shape)
    filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    time_series_filename = [s for s in filenames if shape in s][0]
    time_series_path = os.path.join(data_path, time_series_filename)
    data = np.load(time_series_path)
    train_set += set_to_dicts_list(data, shape, set='training')
    val_set += set_to_dicts_list(data, shape, set='validation')
    test_set += set_to_dicts_list(data, shape, set='testing')

pickle.dump(train_set, open(os.path.join(data_path, 'training.pkl'), "wb"), protocol=2)
pickle.dump(val_set, open(os.path.join(data_path, 'validation.pkl'), "wb"), protocol=2)
pickle.dump(test_set, open(os.path.join(data_path, 'test.pkl'), "wb"), protocol=2)
pickle.dump([train_set, val_set, test_set], open(os.path.join(data_path, 'data_set.pkl'), "wb"), protocol=2)

