import numpy as np
import scipy


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
            print("mean_range")
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
        print("sampling rate: "+str(self.sampling_rate))
        print("sampling freq: "+str(1/self.sampling_rate))

    def set_signal_params(self, **kwargs):

        if self.seq_shape == "constant":
            self.mean_range = kwargs["mean_range"]
        elif self.seq_shape in ["sinusoidal", "square", "sawtooth"]:
            self.freq_range = kwargs["freq_range"]
            self.amp_range = kwargs["amp_range"]
        elif self.seq_shape == "gaussian_pulses":
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
        self.mean_noise_range = kwargs["mean_noise_range"]
        self.deviation_noise_range = kwargs["dev_noise_range"]

    def gen_samples(self, time, seq_shape):
        if seq_shape == "constant":
            value= np.random.uniform(low=self.mean_range[0],
                                     high=self.mean_range[1])
            samples = np.ones(shape=time.shape)*value
            return samples, value
        # TODO: IMPLEMENT OTHER SIGNAL SHAPES

    def generate_dataset(self, **kwargs):
        self.set_prop = kwargs["set_prop"]
        self.n_sequences = kwargs["n_sequences"]



if __name__ == "__main__":
    n_examples = 1000
    set_prop = 0.8, 0.1, 0.1
    data = SequenceGenerator(sequence_shape="sinusoidal",
                             set_prop=set_prop,
                             n_sequences=n_examples)
    # Cadence
    time_sample_noise = 0.3
    max_length = 50
    min_length = 5
    time_span = [10, 50]

    data.set_cadence_params(time_sample_noise=time_sample_noise,
                            max_length=max_length,
                            min_length=min_length,
                            time_span=time_span)

    # Signal
    amp_range = [2, 5]
    freq_range = [0.5, 0.001]

    data.set_signal_params(amp_range=amp_range,
                           freq_range=freq_range)

    # Noise
    heteroskedastic = True
    noise_distr = "gaussian"
    mean_noise = [0.3, 1.7]
    dev_mean = 0.01

    data.set_noise_params(heteroskedastic=heteroskedastic,
                          noise_distr=noise_distr,
                          mean_noise_range=mean_noise,
                          dev_noise_range=dev_mean)


    """Scipy Waveforms
    chirp(t, f0, t1, f1[, method, phi, vertex_zero])	Frequency-swept cosine generator.
    gausspulse(t[, fc, bw, bwr, tpr, retquad, …])	Return a Gaussian modulated sinusoid:
    max_len_seq(nbits[, state, length, taps])	Maximum length sequence (MLS) generator.
    sawtooth(t[, width])	Return a periodic sawtooth or triangle waveform.
    square(t[, duty])	Return a periodic square-wave waveform.
    sweep_poly(t, poly[, phi])	Frequency-swept cosine generator, with a time-dependent frequency.
    unit_impulse(shape[, idx, dtype])	Unit impulse signal (discrete delta function) or unit basis vector."""