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


def sigmoid(spec_amplitude, s, x0):
    return 1. / (1. + np.exp(s * (spec_amplitude - x0)))

def fitting_function(spec_amplitude, sigma, s, x0):
    return rayleigh(spec_amplitude, sigma) * sigmoid(spec_amplitude, s, x0)



def produce_rayleigh_params(bin_centers, histograms, sigma_guess = 0.01, debug=False, percentile=0.9999):
    params = []
    covs = []
    trunc_edges = []
    for i, histogram in enumerate(histograms):
        # if all the bins except for the first are empty this means this is the result of a bandpass filter and fitting is unneccesary
        if np.all(histogram[1:] == 0):
            # param = [0, 0]
            # cov = [[0, 0], [0, 0]]
            trunc_edge=0
            param = [0]
            cov = [[0]]
        else:
            cdf = np.cumsum(histogram*np.diff(bin_centers)[0])
            trunc_edge_idx = np.where(cdf > percentile)[0][0]
            trunc_edge = bin_centers[trunc_edge_idx]
            param, cov  = curve_fit(rayleigh, bin_centers, histogram, p0 = sigma_guess)
            # param, cov  = curve_fit(lambda spec, sigma, s : fitting_function(spec, sigma, s, trunc_edge), bin_centers, histogram, p0 = (sigma_guess, 1e-8))
            if debug:
                if i== 200:
                    plt.stairs(histogram, edges=np.r_[0., bin_centers[:-1] + np.diff(bin_centers), 0.7])
                    # plt.plot(bin_centers, fitting_function(bin_centers, param[0], param[1], trunc_edge))
                    plt.plot(bin_centers, rayleigh(bin_centers, param[0]))
                    plt.vlines(trunc_edge, 0, 1)
                    plt.xlabel("spectral amplitude / V/GHz")
                    plt.ylabel("Counts")
                    plt.yscale("log")
                    plt.ylim(0.00001, None)
                    plt.title(f"{percentile} percentile")
                    plt.show()
                    plt.close()

        params.append(param)
        covs.append(cov)
        trunc_edges.append(trunc_edge)
    return np.squeeze(params), np.squeeze(covs), np.squeeze(trunc_edges)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--comparison_paths", default=None, help="Other spectral histograms to compare distributions, optional", nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = find_config(args.data_path)
    config = read_config(config_path)
    station_id = config["station"]
    if config["skip_clean"]:
        clean = "raw"
    else:
        clean = "clean"


    hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges[0:2])/2


    # real data
    spec_hist_dict = read_pickle(args.data_path)
    frequencies = spec_hist_dict["freq"]
    spec_amplitude_histograms = spec_hist_dict["spec_amplitude_histograms"]


    # convert hists to pdf
    normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
    spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis])
    
    freq_range = [10*units.MHz, 1000*units.MHz]
    selection = np.logical_and(freq_range[0] < frequencies, frequencies < freq_range[1])
    spec_hist_dict["freq"] = frequencies[selection]

    channel_mapping = {"PA" : [0, 1, 2, 3], "PS HPol" : [4, 8], "PS VPol" : [5, 6, 7], "helper Vpol" : [9, 10, 22, 23], "helper HPol" : [11, 21]}

    sigmas_list = []
    covs_list = []
    trunc_edges_list = []
    for ch in range(24):
        print(ch)
        sigmas, covs, trunc_edges = produce_rayleigh_params(bin_centers, spec_amplitude_histograms[ch][selection], debug=args.debug)
        sigmas_list.append(sigmas)
        covs_list.append(covs)
        trunc_edges_list.append(trunc_edges)

    spec_hist_dict["scale_parameters"] = sigmas_list
    spec_hist_dict["scale_parameters_cov"] = covs_list
    spec_hist_dict["trunc_edges"] = trunc_edges_list

    pickle_path = args.data_path.rsplit("_", 1)[0]
    pickle_path += "_scale_params"
    pickle_path += ".pickle"
    print(f"saving as {pickle_path}")
    write_pickle(spec_hist_dict, pickle_path)


    
    if args.debug:
        pdf = PdfPages(f"spec_hist_s{station_id}_sigmas_{clean}_debug.pdf")
        x_lower_lim = 0.15
        x_upper_lim = 0.65
            
        # y_lower_lim = 0.8 * np.min(sigmas_list)
        # y_upper_lim = 7. * np.mean(sigmas_list)


        for i, sigmas in enumerate(sigmas_list):
            covs = covs_list[i]

            fig, ax = plt.subplots()
            ax.errorbar(frequencies[selection], sigmas, yerr = covs, label = "data")
            ax.legend(loc = "best")
            ax.set_title(f"Station{station_id}, channel {i}")
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel(r"$\sigma$")
            ax.set_xlim(x_lower_lim, x_upper_lim)
            # ax.set_ylim(y_lower_lim, y_upper_lim)
        
            fig.savefig(pdf, format="pdf")
            plt.close(fig)
        pdf.close()

        channel_idx = 0
        test_indices = [100, 200, 300, 400, 500]
        fig, axs = plt.subplots(len(test_indices), 1, figsize = (12, 16))
        for i,test_idx in enumerate(test_indices):
            histograms = spec_amplitude_histograms[channel_idx][test_idx]
            axs[i].stairs(histograms, edges=bin_edges)
            axs[i].plot(bin_centers, rayleigh(bin_centers, sigmas_list[channel_idx][test_idx]))
            axs[i].set_title(f"freq = {frequencies[test_idx]:.2f} GHz")
            axs[i].set_yscale("log")
            # ax.text(0.6, 5, f"scale param = {sigmas_list[channel_idx][test_idx]}")
        fig.tight_layout()
        fig.savefig("test")
