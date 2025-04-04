"""
This file should replace 'plot_spec_hist_sigma.py' since
the old file still uses the indices of the datapoints while this file accepts
the histogram directly
"""

import argparse
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy.optimize import curve_fit

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector

from utilities.utility_functions import read_pickle, write_pickle, find_config, read_config
 

def get_config(path):
    config_path = find_config(path)
    config = read_config(config_path)
    return config



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



def get_rayleigh_curve(path):
    config = get_config(path)
    station_id = config["station"]
    if "channels_to_include" in config.keys():
        channels_to_include = config["channels_to_include"]
    else:
        # PA VPol and HPol
        channels_to_include = [0, 4]

    hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges[0:2])/2
    
    spec_hist_dict = read_pickle(path)
    frequencies = spec_hist_dict["freq"]
    spec_amplitude_histograms = spec_hist_dict["spec_amplitude_histograms"]

    # convert hists to pdf
    normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
    spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis])
    
    sigmas_list = []
    covs_list = []
    for ch in channels_to_include:
        sigmas, covs = produce_rayleigh_params(bin_centers, spec_amplitude_histograms[ch], rayleigh)
        sigmas_list.append(sigmas)
        covs_list.append(covs)

    return frequencies, sigmas_list, covs_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    names = ["ice simulation (steffen old antenna)", "ice simulation (masha)", "electronic simulation (140 K)", "data"]

    config = get_config(args.data_paths[0])
    station_id = config["station"]
    
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=station_id,
                            )
    det.update(Time(config["detector_time"]))

    
    frequencies = []
    sigmas_list = []
    covs_list = []
    for data_path in args.data_paths:
        freq, sig, cov = get_rayleigh_curve(data_path)
        frequencies.append(freq)
        sigmas_list.append(sig)
        covs_list.append(cov)
        


    try:
        plt.style.use("~/envs/gaudi.mplstyle")
    except:
        pass

    pdf = PdfPages(f"figures/spec_hist/spec_hist_test.pdf")

    channel_mapping = {"PA" : [0, 1, 2, 3], "PS HPol" : [4, 8], "PS VPol" : [5, 6, 7], "helper Vpol" : [9, 10, 22, 23], "helper HPol" : [11, 21]}

    
    x_lower_lim = 0.15
    x_upper_lim = 0.65
        
    y_lower_lim = 0.8 * np.min(sigmas_list)
    y_upper_lim = 7. * np.mean(sigmas_list)

    
    if "channels_to_include" in config.keys():
        channels_to_include = config["channels_to_include"]
    else:
        # PA VPol and HPol
        channels_to_include = [0, 4]
    
    print(np.array(sigmas_list).shape)

    for ch_i, _ in enumerate(channels_to_include):
        fig, ax = plt.subplots(1, 1, sharex=True)
        for i, sigmas in enumerate(sigmas_list):
            ax.errorbar(frequencies[i], sigmas[ch_i], yerr = covs_list[i][ch_i], label = names[i])
            ax.legend(loc = "best")
            ax.set_title(f"Station{station_id}, channel {channels_to_include[ch_i]})")
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel(r"$\sigma$ / V/GHz")
            ax.set_xlim(x_lower_lim, x_upper_lim)
            ax.set_ylim(y_lower_lim, y_upper_lim)
           
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
