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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--sim_path", default=None)
    parser.add_argument("--force_write", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = find_config(args.data_path)
    print(config_path)
    config = read_config(config_path)
    station_id = config["station"]
    if config["skip_clean"]:
        clean = "raw"
    else:
        clean = "clean"

    config_path = find_config(args.sim_path)
    print(config_path)
    config_sim = read_config(config_path)

    print(config.keys())
    channels_to_include = config_sim["channels_to_include"]
    print(channels_to_include)


    hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges[0:2])/2

    hist_range_sim = config_sim["hist_range"]
    bin_edges_sim = np.linspace(hist_range_sim[0], hist_range_sim[1], nr_bins + 1)
    bin_centers_sim = bin_edges_sim[:-1] + np.diff(bin_edges_sim[0:2])/2
    
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=station_id,
                            )
 
    det.update(Time(config["detector_time"]))

    # plotting data
#    plot_save_path = "data/plotting_data/plot_spec_hist_rayleigh_params_data.pickle"
#    if os.path.exists(plot_save_path) and not args.force_write:
#        plot_dict = read_pickle(plot_save_path)
#        frequencies = plot_dict["frequencies"]
#        sigmas = plot_dict["sigmas"]
#        covs = plot_dict["covs"]
#        sigmas_sim = plot_dict["sigmas_sim"]
#        covs_sim = plot_dict["covs_sim"]


    # real data
    spec_hist_dict = read_pickle(args.data_path)
    frequencies = spec_hist_dict["freq"]
    spec_amplitude_histograms = spec_hist_dict["spec_amplitude_histograms"]

    # simulation
    spec_hist_dict_sim = read_pickle(args.sim_path)
    spec_amplitude_histograms_sim = spec_hist_dict_sim["spec_amplitude_histograms"]


    # convert hists to pdf
    normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
    spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis])
    
    normalization_factor = np.sum(spec_amplitude_histograms_sim, axis = -1) * np.diff(bin_edges_sim)[0]
    spec_amplitude_histograms_sim = np.divide(spec_amplitude_histograms_sim, normalization_factor[:, :, np.newaxis])

    if args.debug:
        channel_index = 0
        freq_index = 300
        sigma_sim, cov_sim = curve_fit(rayleigh, bin_centers_sim, spec_amplitude_histograms_sim[channel_index, freq_index,:], p0=[0.0005])
        fig, ax = plt.subplots()
        ax.stairs(spec_amplitude_histograms_sim[channel_index, freq_index], edges=bin_edges_sim)
        ax.scatter(bin_centers_sim, spec_amplitude_histograms_sim[channel_index, freq_index, :])
        ax.plot(bin_centers_sim, rayleigh(bin_centers_sim, sigma_sim))
        fig.savefig("test")

    try:
        plt.style.use("~/envs/gaudi.mplstyle")
    except:
        pass

    pdf = PdfPages(f"figures/spec_hist/spec_hist_s{station_id}_sigmas_{clean}.pdf")

    channel_mapping = {"PA" : [0, 1, 2, 3], "PS HPol" : [4, 8], "PS VPol" : [5, 6, 7], "helper Vpol" : [9, 10, 22, 23], "helper HPol" : [11, 21]}

    sigmas_list = []
    sigmas_sim_list = []
    covs_list = []
    covs_sim_list = []
    for ch in channels_to_include:
        sigmas, covs = produce_rayleigh_params(bin_centers, spec_amplitude_histograms[ch], rayleigh)
        sigmas_list.append(sigmas)
        covs_list.append(covs)

        if args.sim_path:
            sigmas_sim, covs_sim = produce_rayleigh_params(bin_centers_sim, spec_amplitude_histograms_sim[ch], rayleigh, sigma_guess = 0.00001)  
            sigmas_sim_list.append(sigmas_sim)
            covs_sim_list.append(covs_sim)
    
    x_lower_lim = 0.15
    x_upper_lim = 0.65
        
    y_lower_lim = 0.8 * np.min(sigmas_list)
    y_upper_lim = 7. * np.mean(sigmas_list)

#    for antenna_type, channel_ids in enumerate(channel_mapping.keys()):
#        sigmas = [sigmas_list[channel_id] for channel_id in channel_ids]
#        covs = [covs_list[channel_id] for channel_id in channel_ids]
#        if args.sim_path:
#            sigmas_sim = [sigmas_list_sim[channel_id] for channel_id in channel_ids]
#            covs_sim = [covs_list_sim[channel_id] for channel_id in channel_ids]
#
#        fig, axs = plt.subplots(2, 1, sharex=True)
#        axs[0].errorbar(frequencies, sigmas, yerr = covs, label = "data")
#        axs[0].errorbar(frequencies, sigmas_sim, yerr = covs_sim, label = "simulation")
#        axs[0].legend(loc = "best")
#        axs[0].set_title(f"Station{station_id}, {antenna_type}")
#        axs[0].set_xlabel("freq / GHz")
#        axs[0].set_ylabel(r"$\sigma$")
#        axs[0].set_ylim(y_lower_lim, y_upper_lim)


    for i, sigmas in enumerate(sigmas_list):
        covs = covs_list[i]
        if args.sim_path:
            sigmas_sim = sigmas_sim_list[i]
            covs_sim = covs_sim_list[i]

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].errorbar(frequencies, sigmas, yerr = covs, label = "data")
        sim_ax = axs[0].twinx()
        sim_ax.errorbar(frequencies, sigmas_sim, yerr = covs_sim, label = "simulation", color = "red")
        print(sigmas_sim)
        sim_ax.legend()
        axs[0].legend(loc = "best")
        axs[0].set_title(f"Station{station_id}, channel {channels_to_include[i]}")
        axs[0].set_xlabel("freq / GHz")
        axs[0].set_ylabel(r"$\sigma$")
        axs[0].set_xlim(x_lower_lim, x_upper_lim)
        axs[0].set_ylim(y_lower_lim, y_upper_lim)
       
        if args.sim_path:
            ratio = np.divide(sigmas, sigmas_sim, out=np.zeros_like(sigmas), where=sigmas>np.min(sigmas))
            ratio_mean = np.mean(ratio[np.where(sigmas>np.min(sigmas))])
            axs[1].plot(frequencies, ratio)
            axs[1].set_xlabel("freq / GHz")
            axs[1].set_ylabel("ratio (data / sim)")
            axs[1].text(0.95, 0.95, f"max ratio = {np.max(ratio):.2f} \n mean ratio = {ratio_mean:.2f}",
                        size="small",
                        transform=axs[1].transAxes, ha="right", va="top", backgroundcolor="lightblue")

        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
