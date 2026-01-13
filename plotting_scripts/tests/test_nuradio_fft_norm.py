
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import fft, units, constants


from utilities.utility_functions import read_freq_spectrum_from_nur, read_freq_spectrum_from_pickle




if __name__ == "__main__":

    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    impedance = 50 * units.ohm
    freqs = fft.freqs(nr_samples, sampling_rate)

    test_spectrum = np.ones_like(freqs)
    test_trace = fft.freq2time(test_spectrum, sampling_rate)
    vrms_test_trace = np.sqrt(np.mean(test_trace**2))
    print(f"vrms test trace is {vrms_test_trace}")

    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()
    generated_spectrum = generic_noise_adder.bandlimited_noise(100*units.MHz, 600*units.MHz, nr_samples, sampling_rate, vrms_test_trace, time_domain=False)
    print(np.abs(generated_spectrum))
    generated_trace = generic_noise_adder.bandlimited_noise(100*units.MHz, 600*units.MHz, nr_samples, sampling_rate, vrms_test_trace, time_domain=True)
    vrms_generated_trace = np.sqrt(np.mean(generated_trace**2))
    print(f"vrms generated trace is {vrms_generated_trace}")


    # for Rayleigh

    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()
    mean_spectrum = np.zeros_like(freqs)
    nr_sims = 10000
    for i in range(nr_sims):
        generated_spectrum = generic_noise_adder.bandlimited_noise(100*units.MHz, 600*units.MHz, nr_samples, sampling_rate, vrms_test_trace, type="rayleigh", time_domain=False)
        mean_spectrum += np.abs(generated_spectrum)**2
    mean_spectrum /= nr_sims
    mean_spectrum = np.sqrt(mean_spectrum)
    mean_trace = fft.freq2time(mean_spectrum, sampling_rate)
    vrms_mean = np.sqrt(np.mean(mean_trace**2))
    print(f"vrms generated trace is {vrms_mean}")



    # testing mean hypothesis

    path_to_test_data = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/season2023/station11/clean/spectra_run1935.nur"
    spectra = read_freq_spectrum_from_nur(path_to_test_data)
    channel_id = 0

    test_spectra = spectra[:,channel_id]
    test_traces = [fft.freq2time(spec, sampling_rate) for spec in test_spectra]

    mean_vrms_from_traces = np.mean([np.sqrt(np.mean(t**2)) for t in test_traces])

    print(mean_vrms_from_traces)


    mean_spectra = np.sqrt(np.mean(test_spectra**2, axis=0))
    meaned_trace = fft.freq2time(mean_spectra, sampling_rate)
    vrms_from_meaning_of_spectra = np.sqrt(np.mean(meaned_trace**2))
    print(vrms_from_meaning_of_spectra)

    freq_index = 200
    print(freqs[freq_index])

    plt.hist(test_spectra[:][freq_index], bins=100)
    plt.vlines(mean_spectra[freq_index], 0, 100, color="red")
    plt.savefig("bla")


    test_data = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/job_2025_11_04_test/station11/clean/average_ft_run2305.pickle"
    new_spectra = read_freq_spectrum_from_pickle(test_data)
    print(new_spectra["var_spectrum"][0])
    new_spectra = np.array(new_spectra["spectrum"])
    print(new_spectra.shape)
    new_trace = fft.freq2time(new_spectra[channel_id], sampling_rate)
    print(np.sqrt(np.mean(new_trace**2)))

