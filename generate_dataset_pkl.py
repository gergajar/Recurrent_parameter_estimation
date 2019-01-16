import os
import sys
import numpy as np

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ''))
sys.path.append(PATH_TO_PROJECT)
from sequence_generator import SequenceGenerator
import pickle
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    #delete current pkls
    data_path = os.path.join(PATH_TO_PROJECT, 'data')
    filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    pkl_filenames = [s for s in filenames if '.pkl' in s]
    for pkl_file in pkl_filenames:
        path_to_file = os.path.join(data_path, pkl_file)
        if os.path.isfile(path_to_file):
            os.remove(path_to_file)

    shapes = ["square", "sawtooth", "sinusoidal"]#["sinusoidal"]#
    for shape in shapes:

        data = SequenceGenerator(sequence_shape=shape)
        # Cadence
        time_sample_noise = 0.0
        max_length = 100#50
        min_length = 100#20
        time_span = [0,3]#[10, 50]
        min_time_spam = 3#10

        data.set_cadence_params(time_sample_noise=time_sample_noise,
                                max_length=max_length,
                                min_length=min_length,
                                time_span=time_span,
                                min_time_spam=min_time_spam)

        # Signal
        amp_range = [0.4, 1] #[0.6, 0.8]
        #period_range = [np.pi/2, np.pi/2]
        #periods = np.linspace(start=3/4, stop=3, num=8)#np.linspace(start=np.pi/2, stop=2*np.pi, num=4) #[3/4]
        periods_range = [3/4, 3]
        freq_range = 1/np.array(periods_range)[::-1]#np.array(period_range)/(2*np.pi)#freq_range = [0.3, 0.05]
        #freqs = 1/np.array(periods)#(2 * np.pi)/np.array(periods)

        data.set_signal_params(amp_range=amp_range,
                               freqs = None,#freqs,
                               freq_range=freq_range)

        # Noise
        heteroskedastic = True
        noise_distr = "gaussian"
        mean_noise = [0.01, 0.2]
        dev_mean = 0.01
        amp_noise = 0.0
        phase_noise = 0#2*np.pi

        data.set_noise_params(heteroskedastic=heteroskedastic,
                              noise_distr=noise_distr,
                              noise_range=mean_noise,
                              dev_noise_range=dev_mean,
                              amp_noise=amp_noise,
                              phase_noise=phase_noise)

        n_examples = 1000#37500#100#100000#37500
        set_prop = 0.8, 0.1, 0.1

        data.generate_dataset(set_prop=set_prop,
                              n_sequences=n_examples,
                              data_name=shape)

        #data.plot_n_examples()

    #SCRIPT2 sequences to dicts
    # kwargs["amp"], kwargs["freq"], kwargs["phase"]
    def set_to_dicts_list(data, time_series_class, set):
        data = data[set]
        dicts_list = []
        for sequence_index in range(data['n_sequences']):
            sequence_dict = {'original_magnitude': data['real_values'][sequence_index],
                             'generated_magnitude': data['sequences'][sequence_index],
                             'original_time': data['dense_time'][sequence_index],
                             'frequency': data['params'][sequence_index][1],
                             'amplitude': data['params'][sequence_index][0],
                             'phase': data['params'][sequence_index][2],
                             'class': time_series_class, }
            dicts_list.append(sequence_dict)
        return dicts_list


    data_path = os.path.join(PATH_TO_PROJECT, 'data')
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
    pickle.dump([train_set, val_set, test_set], open(os.path.join(data_path, 'data_set_dicts.pkl'), "wb"),
                protocol=2)

    del train_set
    del val_set
    del test_set

    #SCRIPT3 concatenate dicts
    data_path = os.path.join(PATH_TO_PROJECT, 'data')
    # pickle.dump([train_set, val_set, test_set], open(os.path.join(data_path, 'data_set_dicts.pkl'), "wb"), protocol=2)
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
    print('train shape %s' % str(np.array(train['original_magnitude']).shape))
    print('val shape %s' % str(np.array(val['original_magnitude']).shape))
    print('test shape %s' % str(np.array(test['original_magnitude']).shape))