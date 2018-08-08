import numpy as np
import pickle


class SequenceDataset(object):

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.data_path = kwargs["data_path"]
        self.n_inputs = kwargs["n_inputs"]
        self.input_data_path = kwargs["input_data_path"]
        self.rebuild_data = kwargs["rebuild_data"]

        self.current_batch = {"training": 0, "validation": 0, "testing": 0}
        self.current_epoch = {"training": 0, "validation": 0, "testing": 0}

        self.data = np.load(self.data_path)
        print(list(self.data.keys()))
        print(list(self.data["training"]))

        if self.rebuild_data:
            self.build_network_input()

        self.network_input = np.load(self.input_data_path)
        self.n_batches_subset = {}
        for subset_name in self.current_batch.keys():
            self.n_batches_subset[subset_name] = len(self.network_input[subset_name])
            print("n_batches " + subset_name, len(self.network_input[subset_name]))

    def build_network_input(self):

        def build_input_shape(data_array, input_length, max_length):
            shaped_input = []
            first = 0
            last = input_length
            while last < input_length:
                shaped_input.append(data_array[first:last][np.newaxis, ...])
                first += 1
                last += 1
            seq_length = len(shaped_input)
            n_fill = max_length - len(shaped_input)
            shaped_input.append(np.zeros(shape=(n_fill, input_length)))
            shaped_input = np.concatenate(shaped_input, axis=0)
            return shaped_input, seq_length

        input_to_save = {}

        for subset in self.data.keys():
            max_length = np.amax(self.data[subset]["max_length"])
            n_sequences = self.data[subset]["n_sequences"]
            keys_to_convert = ["sequences", "noise", "time"]
            start = 0
            end = self.batch_size
            batch_list = []
            while end < n_sequences:
                batch_dict = {"sequences": [],
                              "noise": [],
                              "time": [],
                              "lengths": [],
                              "params": [],
                              "index": []}
                for i in np.arange(start=start, stop=end, step=1):
                    for key in keys_to_convert:
                        converted, seq_l = build_input_shape(self.data[subset][key][i], self.n_inputs, max_length)
                        batch_dict[key].append(converted)
                    batch_dict["params"].append(self.data[subset][key][i])
                    batch_dict["lengths"].append(seq_l)
                    batch_dict["index"].append(i)
                for key in batch_dict.keys():
                    batch_dict[key] = np.array(batch_dict[key])
                batch_list.append(batch_dict)
                if end == n_sequences:
                    break
                start += self.batch_size
                end += self.batch_size
                if end >= n_sequences:
                    end = n_sequences
            input_to_save[subset] = batch_list

        pickle.dump(input_to_save,
                    open(self.input_data_path, "wb"),
                    protocol=2)

    def next_batch(self, subset_name="training"):
        batch = self.network_input[subset_name][self.current_batch[subset_name]]
        self.current_batch[subset_name] += 1
        if self.current_batch[subset_name] >= self.n_batches_subset[subset_name]:
            self.current_batch[subset_name] = 0
            self.current_epoch[subset_name] += 1
        return batch


if __name__ == "__main__":

    data_path = "./data/sinusoidal.pkl"
    input_data_path = "./data/network_input.pkl"
    n_inputs = 1
    rebuild_data = True
    dataset = SequenceDataset(data_path=data_path,
                              batch_size=128,
                              n_inputs=1,
                              input_data_path=input_data_path,
                              rebuild_data=rebuild_data)

    n_iter = 10
    for i in range(n_iter):
        batch = dataset.next_batch("training")