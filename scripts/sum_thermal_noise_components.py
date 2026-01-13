import argparse
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os

from NuRadioReco.utilities import fft

from utilities.utility_functions import read_freq_spectrum_from_nur, write_pickle




def combine_times(begin_time1, end_time1, begin_time2, end_time2):
    if begin_time1 < begin_time2:
        begin_time = begin_time1
    else:
        begin_time = begin_time2
    if end_time1 > end_time2:
        end_time = end_time1
    else:
        end_time = end_time2
    return begin_time, end_time



if __name__ == "__main__":

    
    # SETTINGS
    station_id = 11
    nr_samples = 2048
    sampling_rate = 3.2
    channel_ids = np.arange(24)
    test = False


    source_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/digitizer_v2/station{station_id}"
    sim_components = [
            glob.glob(f"{source_dir}/run*/events_ice_batch*.nur"),
            glob.glob(f"{source_dir}/run*/events_electronic_batch*.nur"),
            glob.glob(f"{source_dir}/run*/events_galactic_batch*.nur"),
            ]

    sim_components = np.array([natsorted(comp) for comp in sim_components]).T
    if test:
        sim_components = sim_components[:2]

    freqs = fft.freqs(nr_samples, sampling_rate)
    mean_spectra = np.zeros((len(channel_ids), len(freqs)))
    squared_spectra = np.zeros((len(channel_ids), len(freqs)))
    nr_events = 0
    for i, component_batch in enumerate(sim_components):
        print(i)
        ice_spectra = read_freq_spectrum_from_nur(component_batch[0], return_phase=True)
        electronic_spectra = read_freq_spectrum_from_nur(component_batch[1], return_phase=True)
        galactic_spectra = read_freq_spectrum_from_nur(component_batch[2], return_phase=True)
        nr_events += len(ice_spectra)

        ice_traces = fft.freq2time(ice_spectra, sampling_rate)
        electronic_traces = fft.freq2time(electronic_spectra, sampling_rate)
        galactic_traces = fft.freq2time(galactic_spectra, sampling_rate)

        summed_traces = ice_traces + electronic_traces + galactic_traces
        summed_spectra =  fft.time2freq(summed_traces, sampling_rate)

        mean_spectra = mean_spectra + np.sum(np.abs(summed_spectra)**2, axis=0)
        squared_spectra = squared_spectra + np.sum(np.abs(summed_spectra**4), axis=0)

        if test and i == len(sim_components) - 1:
            plt.plot(fft.freqs(nr_samples, sampling_rate), np.sqrt((np.mean(np.abs(summed_spectra)**2, axis=0)))[0])
            plt.xlim(0., 1.)
            plt.savefig("test_summing_sim_comps")


    mean_spectra /= nr_events
    mean_spectra = np.sqrt(mean_spectra)
    squared_spectra /= nr_events
    var_spectra = squared_spectra - mean_spectra
    var_spectra /= nr_events
    # delta method
    var_spectra = (1/(4*mean_spectra**2)) * var_spectra
    
    

    result_dictionary = {}
    result_dictionary["freq"] = freqs
    result_dictionary["frequency_spectrum"] = mean_spectra
    result_dictionary["var_frequency_spectrum"] = var_spectra
    result_dictionary["header"] = {}
    result_dictionary["header"]["nr_events"] = nr_events


    saved_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/testing_comp_sum"
    pickle_file = f"{saved_dir}/station{station_id}/clean"
    if not os.path.exists(pickle_file):
        os.makedirs(pickle_file)

    pickle_file += "/average_ft_summed_comps.pickle"
    write_pickle(result_dictionary, pickle_file)
