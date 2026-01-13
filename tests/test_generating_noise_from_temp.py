import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import fft, units
from NuRadioReco.utilities.signal_processing import calculate_vrms_from_temperature



if __name__ == "__main__":
    
    test_temp = 80 * units.kelvin
    bandwidth = [0.1, 0.7]
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz


    test_vrms = calculate_vrms_from_temperature(test_temp, bandwidth=bandwidth)

    freqs = fft.freqs(nr_samples, sampling_rate)
    noise_adder = channelGenericNoiseAdder()
    noise = noise_adder.bandlimited_noise(bandwidth[0], bandwidth[-1], nr_samples, sampling_rate, test_vrms, time_domain=False)

    plt.plot(freqs, np.abs(noise))

    bandwidth = [0., 1.6]
    test_vrms = calculate_vrms_from_temperature(test_temp, bandwidth=bandwidth)

    freqs = fft.freqs(nr_samples, sampling_rate)
    noise_adder = channelGenericNoiseAdder()
    noise = noise_adder.bandlimited_noise(bandwidth[0], bandwidth[-1], nr_samples, sampling_rate, test_vrms, time_domain=False)
    print(np.abs(noise))
    plt.plot(freqs, np.abs(noise))
    plt.savefig("test")
