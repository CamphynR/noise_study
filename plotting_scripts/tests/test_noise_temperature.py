import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft
from NuRadioReco.utilities import fft, units


from utilities.utility_functions import read_freq_spectrum_from_nur




if __name__ == "__main__":

    path_to_test_data = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/season2023/station11/clean/spectra_run1930.nur"
    
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    impedance = 50 * units.ohm
    freqs = fft.freqs(nr_samples, sampling_rate)
    spectra = read_freq_spectrum_from_nur(path_to_test_data)

    test_spectrum = spectra[0][0]
    test_trace = fft.freq2time(test_spectrum, sampling_rate)
    scipy_freqs = scipy_fft.fftfreq(nr_samples, d=1./sampling_rate)
    scipy_test_spectrum = scipy_fft.fft(test_trace)

    print(f"vrms trace is {np.sqrt(np.mean(test_trace**2))} V")
    print(f"vrms spectrum is {np.sqrt(np.sum(test_spectrum**2) * np.diff(freqs)[0] * sampling_rate/nr_samples)} V")
    print(f"vrms scipy spectrum is {np.sqrt(np.sum(np.abs(scipy_test_spectrum)**2*np.diff(scipy_freqs)[0])*sampling_rate/nr_samples)} V")

    plt.plot(freqs, spectra[0][0])
    plt.savefig("test_noise_temp")

