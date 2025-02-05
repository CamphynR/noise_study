"""
This file should replace 'plot_spec_hist_sigma.py' since
the old file still uses the indices of the datapoints while this file accepts
the histogram directly

This version does not plot simulations and plots channels grouped together
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
from utilities.temp_to_noise import temp_to_volt
 



def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))



def produce_rayleigh_params(bin_centers, histograms, rayleigh_function):
    params = []
    covs = []
    for i, histogram in enumerate(histograms):
        param, cov  = curve_fit(rayleigh_function, bin_centers, histogram, p0 = [0.01])
        params.append(param)
        covs.append(cov)
    return np.squeeze(params), np.squeeze(covs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    args = parser.parse_args()

    config_path = find_config(args.data_path)
    config = read_config(config_path)
    station_id = config["station"]
    if config["skip_clean"]:
        clean = "raw"
    else:
        clean = "clean"

    print("started")

    hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges[0:2])/2
    
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


    # convert hists to pdf
    normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
    spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis])
    


    try:
        plt.style.use("~/envs/gaudi.mplstyle")
    except:
        pass

    pdf_name = f"figures/spec_hist/spec_hist_s{station_id}_sigmas_{clean}_channels_grouped.pdf"

    pdf = PdfPages(pdf_name)

    channel_mapping = {"PA" : [0, 1, 2, 3], "PS HPol" : [4, 8], "PS VPol" : [5, 6, 7], "helper Vpol" : [9, 10, 22, 23], "helper HPol" : [11, 21]}

    sigmas_list = []
    covs_list = []
    for ch in range(24):
        sigmas, covs = produce_rayleigh_params(bin_centers, spec_amplitude_histograms[ch], rayleigh)
        sigmas_list.append(sigmas)
        covs_list.append(covs)
    sigmas_list = np.array(sigmas_list)
    covs_list = np.array(covs_list)
    
    x_lower_lim = 0.15
    x_upper_lim = 0.65
        
    y_lower_lim = 0.8 * np.min(sigmas_list)
    y_upper_lim = 0.21
#    y_upper_lim = 7. * np.mean(sigmas_list)

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


    for channel_name in channel_mapping:
        channel_idxs = channel_mapping[channel_name]
        print(channel_idxs)
        print(channel_name)

        fig, ax = plt.subplots()
        sigmas = sigmas_list[channel_idxs]
        covs = covs_list[channel_idxs]
        for i, sigma in enumerate(sigmas):
            ax.errorbar(frequencies, sigmas[i], yerr = covs[i], label = f"channel {channel_idxs[i]}")
            ax.set_title(f"Station{station_id}, {channel_name}")
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel(r"$\sigma$ / V")
            ax.set_xlim(x_lower_lim, x_upper_lim)
            ax.set_ylim(y_lower_lim, y_upper_lim)
           
        ax.legend(loc = "best")

        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches = "tight", dpi = 400)
        plt.close(fig)
    pdf.close()
    print(f"saved as {pdf_name}")
