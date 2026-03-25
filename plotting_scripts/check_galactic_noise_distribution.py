import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import freqs

from utilities.utility_functions import read_freq_spectrum_from_nur


def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))



def produce_rayleigh_params(bin_centers, histograms, rayleigh_function, sigma_guess = 0.01):
    params = []
    covs = []
    for i, histogram in enumerate(histograms):
        # if all the bins except for the first are empty this means this is the result of a bandpass filter and fitting is unneccesary
        if np.all(histogram[1:] == 0):
            param = [0]
            cov = [[0]]
        else:
            param, cov  = curve_fit(rayleigh_function, bin_centers, histogram, p0 = sigma_guess)
        params.append(param)
        covs.append(cov)
    return np.squeeze(params), np.squeeze(covs)



if __name__ == "__main__":

    station_id = 11
    nr_channels = 24

    galatic_paths = glob.glob("/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.2_no_system_response_measured_electronic_noise/digitizer_v2/station11/run*/events_galactic_batch*.nur")

    nr_bins = 30
    hist_range = [0, 1e-4]

    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centres = bin_edges[:-1] + np.diff(bin_edges[0:2]) / 2

    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = freqs(nr_samples, sampling_rate) 


    distributions = np.zeros((nr_channels, len(frequencies), nr_bins))

    for path in galatic_paths[:5]:
        spectra = read_freq_spectrum_from_nur(path)
        for spectrum in spectra:
            for channel_id in range(nr_channels):
                spectrum_channel = spectrum[channel_id]
                bin_indices = np.searchsorted(bin_edges[1:-1], spectrum_channel)
                distributions[channel_id, np.arange(len(frequencies)), bin_indices] += 1



    channel_id = 0
    frequency = 300 * units.MHz
    freq_index = np.where(np.isclose(frequency, frequencies, atol=np.diff(frequencies)[0]/2.))[0][0]

    normalization_factor = np.sum(distributions, axis = -1) * np.diff(bin_edges)[0]
    distributions = np.divide(distributions, normalization_factor[:, :, np.newaxis])

    distribution = distributions[channel_id][freq_index]

    sigma, cov = curve_fit(rayleigh, bin_centres, distributions[channel_id, freq_index,:], p0=[0.000005])
    

    plt.style.use("retro")
    plt.bar(bin_centres, distribution, width=0.95*np.diff(bin_edges))
    plt.plot(bin_centres, rayleigh(bin_centres, sigma), color="red")
    plt.savefig("test")
