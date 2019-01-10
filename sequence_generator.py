import numpy as np
import scipy.signal as ss
import pickle
import matplotlib.pyplot as plt

np.random.seed(0)

class SequenceGenerator(object):

    def __init__(self, **kwargs):

        print("Available shapes: ")
        print("constant, sinusoidal, square, sawtooth, gaussian_pulses")
        print("Frequency modulated: ")
        print("mod_sin, mod_square, mod_sawtooth")
        self.seq_shape = kwargs["sequence_shape"]
        print("Selected shape: "+self.seq_shape)
        print("Shape parameters: ")
        if self.seq_shape == "constant":
            print("amp_range")
        elif self.seq_shape in ["sinusoidal", "square", "sawtooth"]:
            print("freq_range")
            print("amp_range")
        elif self.seq_shape == "gaussian_pulses":
            print("freq_range")
            print("amp_range")
            print("pulse_width_range")

    def set_cadence_params(self, **kwargs):
        """
        params:
        time_sample_noise: float, gaussian deviation from regular sampling
        max_length: int, max number of points within a sequence
        min_length: int, min number of points within a sequence
        time_span: [t0, tf] time_range to sample
        """
        self.time_sample_noise = kwargs["time_sample_noise"]
        self.max_length = kwargs["max_length"]
        self.min_length = kwargs["min_length"]
        self.time_span = kwargs["time_span"]
        self.sampling_rate = self.time_span[-1]/self.max_length
        self.min_time_spam = kwargs["min_time_spam"]
        print("sampling rate: "+str(self.sampling_rate))
        print("sampling freq: "+str(1/self.sampling_rate))

    def set_signal_params(self, **kwargs):

        if self.seq_shape == "constant":
            self.amp_range = kwargs["amp_range"]
        elif self.seq_shape in ["sinusoidal", "square", "sawtooth"]:
            self.freq_range = kwargs["freq_range"]
            self.freqs = kwargs["freqs"]
            self.amp_range = kwargs["amp_range"]
        elif self.seq_shape == "gaussian_pulses":
            self.freqs = kwargs["freqs"]
            self.freq_range = kwargs["freq_range"]
            self.amp_range = kwargs["amp_range"]
            self.pulse_width_range = kwargs["pulse_width_range"]

    def set_noise_params(self, **kwargs):
        """
        params:
        noise_distr: uniform, gaussian, poisson
        heteroskedastic: True is heteroskedastic noise
        mean_noise_range: float, range of mean noise
        dev_noise_range: float, std from sampled mean noise
        """
        self.noise_distr = kwargs["noise_distr"]
        self.heteroskedastic = kwargs["heteroskedastic"]
        self.noise_range = kwargs["noise_range"]
        self.deviation_noise_range = kwargs["dev_noise_range"]

    def gen_samples(self, time, seq_shape, kwargs):
        # TODO: MAKE IT DEPENDENT OF PARAMETERS, SAMPLE OUTSIDE
        if seq_shape == "constant":
            samples = np.ones(shape=time.shape)*kwargs["amp"]
            params = np.array([kwargs["amp"], ])
            return samples, params
        elif seq_shape in ["sinusoidal", "square", "sawtooth"]:
            samples = 0
            if seq_shape == "sinusoidal":
                samples = np.sin(2*np.pi*kwargs["freq"]*time + kwargs["phase"])*kwargs["amp"]
            elif seq_shape == "square":
                samples = ss.square(2*np.pi*kwargs["freq"]*time + kwargs["phase"])*kwargs["amp"]
            elif seq_shape == "sawtooth":
                samples = ss.sawtooth(2 * np.pi * kwargs["freq"] * time + kwargs["phase"])*kwargs["amp"]
            params = np.array([kwargs["amp"], kwargs["freq"], kwargs["phase"]])
            return samples, params

        # elif seq_shape == "gaussian_pulses":
        #    amp = np.random.uniform(low=self.amp_range[0],
        #                            high=self.amp_range[1])
        #    freq = np.random.uniform(low=self.freq_range[0],
        #                             high=self.freq_range[1])
        #    pulse_width = np.random.uniform(low=self.pulse_width_range[0],
        #                                    high=self.pulse_width_range[1])

    def generate_sequences(self, n_sequences):

        sequence_list = []
        time_list = []
        params_list = []
        noise_list = []
        real_values_list = []
        dense_time_list = []
        sequence_length_list = []
        for i in range(n_sequences):
            samples_within_sequence = np.random.uniform(low=self.min_length,
                                                        high=self.max_length+1)
            last_sample_time = np.random.uniform(low=self.time_span[0]+self.min_time_spam, high=self.time_span[1])
            amp = np.random.uniform(low=self.amp_range[0],
                                    high=self.amp_range[1])
            #d noise of giorgia
            #amp += np.random.normal(loc=0, scale=0.1)
            if self.freq_range is not None:
                freq = np.random.uniform(low=self.freq_range[0],
                                         high=self.freq_range[1])
            else:
                freq = np.random.choice(self.freqs)
            #No random phase
            #phase = np.random.uniform(low=0, high=2*np.pi)
            phase = 0
            kwargs = {"amp": amp, "freq": freq, "phase": phase}

            time = np.linspace(start=self.time_span[0], stop=last_sample_time, num=samples_within_sequence)
            sequence_length_list.append(len(time))
            time += np.random.normal(loc=0, scale=self.time_sample_noise, size=time.size)
            time = np.sort(time)
            dense_time = np.linspace(start=self.time_span[0], stop=last_sample_time, num=samples_within_sequence)
            #np.arange(start=self.time_span[0], stop=last_sample_time, step=0.1)
            underlying_model, _ = self.gen_samples(time=dense_time, seq_shape=self.seq_shape, kwargs=kwargs)
            samples, params = self.gen_samples(time=time, seq_shape=self.seq_shape, kwargs=kwargs)
            real_values_list.append(underlying_model)
            dense_time_list.append(dense_time)

            if self.heteroskedastic:
                if self.noise_distr == "gaussian":
                    noise_within_sequence = np.random.uniform(low=self.noise_range[0],
                                                              high=self.noise_range[1],
                                                              size=samples.size)
                    noise_list.append(noise_within_sequence)
                    samples += np.random.normal(scale=noise_within_sequence)
                elif self.noise_distr == "poisson":
                    noise_list.append(np.sqrt(np.abs(samples)))
                    samples += np.multiply(np.random.poisson(lam=np.abs(samples)), np.sign(samples))

            else:
                noise_within_sequence = np.random.uniform(low=self.noise_range[0],
                                                          high=self.noise_range[1])
                samples += np.random.normal(scale=noise_within_sequence, size=samples.size)
                noise_list.append(noise_within_sequence)

            sequence_list.append(samples)
            time_list.append(time)
            params_list.append(params)

        params_list = np.stack(params_list)
        sequence_length_list = np.stack(sequence_length_list)

        return sequence_list, real_values_list, time_list, params_list, noise_list, dense_time_list, sequence_length_list

    def generate_dataset(self, **kwargs):

        self.set_prop = np.array(kwargs["set_prop"])
        self.n_sequences = kwargs["n_sequences"]
        self.data_name = kwargs["data_name"]
        self.data_path = "./data/"

        sets_names = ["training", "validation", "testing"]
        datasets = {}
        for set_i, name in enumerate(sets_names):
            datasets[name] = {}
            datasets[name]["n_sequences"] = np.round(self.set_prop[set_i]*self.n_sequences).astype(np.int)
            print(name, "n_sequences: ", datasets[name]["n_sequences"])
            sequences, real_values, times, params, noise, dense_time, lengths = self.generate_sequences(datasets[name]["n_sequences"])
            datasets[name]["sequences"] = sequences
            datasets[name]["real_values"] = real_values
            datasets[name]["time"] = times
            datasets[name]["params"] = params
            datasets[name]["noise"] = noise
            datasets[name]["dense_time"] = dense_time
            datasets[name]["lengths"] = lengths
            datasets[name]["max_length"] = self.max_length

        pickle.dump(datasets,
                    open(self.data_path + self.data_name + "_" + self.noise_distr + ".pkl", "wb"),
                    protocol=2)

    def plot_n_examples(self, subset_name="training", n_examples=3, data_path=None):

        if not data_path:
            data_path = self.data_path + self.data_name + "_" + self.noise_distr + ".pkl"

        data = np.load(data_path)
        subset = data[subset_name]
        for i in range(n_examples):
            plt.plot(subset["dense_time"][i], subset["real_values"][i], label="underlying_model")
            plt.errorbar(subset["time"][i], subset["sequences"][i], yerr=subset["noise"][i], fmt="o", ms=5, label="samples")
            plt.xlabel("time")
            plt.ylabel("amplitude")
            title = "amp: "+"{0:.2f}".format(subset["params"][i][0])\
                    +", freq: "+"{0:.2f}".format(subset["params"][i][1])\
                    +", period: "+"{0:.2f}".format(1/subset["params"][i][1])
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.savefig("./plots/"+subset_name+"_"+self.seq_shape+
                        "_"+self.noise_distr+"_sample"+str(i).zfill(3)+".png")
            plt.show()
            #plt.close("all")


if __name__ == "__main__":

    shapes = ["square", "sawtooth", "sinusoidal"]
    for shape in shapes:

        data = SequenceGenerator(sequence_shape=shape)
        # Cadence
        time_sample_noise = 0.7
        max_length = 100#50
        min_length = 100#20
        time_span = [0,12]#[10, 50]
        min_time_spam = 12#10

        data.set_cadence_params(time_sample_noise=time_sample_noise,
                                max_length=max_length,
                                min_length=min_length,
                                time_span=time_span,
                                min_time_spam=min_time_spam)

        # Signal
        amp_range = [1, 1]
        #period_range = [np.pi/2, np.pi/2]
        periods = np.linspace(start=np.pi/2, stop=2*np.pi, num=10)
        #freq_range = np.array(period_range)/(2*np.pi)#freq_range = [0.3, 0.05]
        freqs = (2 * np.pi)/np.array(periods)

        data.set_signal_params(amp_range=amp_range,
                               freqs = freqs,
                               freq_range=None)

        # Noise
        heteroskedastic = True
        noise_distr = "gaussian"
        mean_noise = [0.3, 1.7]
        dev_mean = 0.01

        data.set_noise_params(heteroskedastic=heteroskedastic,
                              noise_distr=noise_distr,
                              noise_range=mean_noise,
                              dev_noise_range=dev_mean)

        n_examples = 37500
        set_prop = 0.8, 0.1, 0.1

        data.generate_dataset(set_prop=set_prop,
                              n_sequences=n_examples,
                              data_name=shape)

        data.plot_n_examples()


    """Scipy Waveforms
    chirp(t, f0, t1, f1[, method, phi, vertex_zero])	Frequency-swept cosine generator.
    gausspulse(t[, fc, bw, bwr, tpr, retquad, …])	Return a Gaussian modulated sinusoid:
    max_len_seq(nbits[, state, length, taps])	Maximum length sequence (MLS) generator.
    sawtooth(t[, width])	Return a periodic sawtooth or triangle waveform.
    square(t[, duty])	Return a periodic square-wave waveform.
    sweep_poly(t, poly[, phi])	Frequency-swept cosine generator, with a time-dependent frequency.
    unit_impulse(shape[, idx, dtype])	Unit impulse signal (discrete delta function) or unit basis vector."""