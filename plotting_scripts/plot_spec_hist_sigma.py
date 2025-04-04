import argparse
import glob
import json
import os
from pathlib import Path
import pickle
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


from utility_functions import open_pickle, find_config


def read_hist_from_data_dir(data_dir, nr_bins):
    channels = np.arange(24)
    run_files = glob.glob(f"{data_dir}/*.pickle")
    nr_runs = len(run_files)
    base = open_pickle(run_files[0])
    base = base["var"]
    freqs = base[0][0][0]
    sampling_rate = base[0][0][1]
    spec_hist = np.zeros((len(channels), len(freqs), nr_bins))
    for run_file in run_files:
        hist_output = open_pickle(run_file)
        hist_output = hist_output["var"]
        # output shape is [events, channels, 3], where 3 is freqs, bin_idxs or sampling_rate
        for hist_output_event in hist_output:
            for channel in range(len(channels)):
                for freq_idx in range(len(freqs)):
                    if hist_output_event[channel][1][freq_idx] == nr_bins:
                        continue
                    else:
                        spec_hist[channel, freq_idx, hist_output_event[channel][1][freq_idx]] += 1
    # spec_hists shape is (channels, frequencies, hists)

    return freqs, spec_hist, sampling_rate, nr_runs


def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-dir", help="Directory containing run pickles (should be named clean or raw)")
    args = parser.parse_args()
    station_nr = int(str(Path(args.data_dir).parents[0]).split("/")[-1][-2:])
    clean = args.data_dir.split("/")[-1]

    config_path = find_config(args.data_dir)
    print(config_path)
    with open(config_path) as config_file:
        config = json.load(config_file)

    hist_range = config["variable_function_kwargs"][f"{clean}"]["range"]
    nr_bins = config["variable_function_kwargs"][f"{clean}"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    pickle_path = f"data/plotting_data/spec_hist/spec_hist_s{station_nr}_{clean}.pickle"
    if os.path.exists(pickle_path):
        pickle_dir = open_pickle(pickle_path)
        freqs = pickle_dir["freqs"]
        spec_hists = pickle_dir["spec_hists"]
        sampling_rate = pickle_dir["sampling_rate"]
        nr_runs = pickle_dir["nr_runs"]
    else:
        freqs, spec_hists, sampling_rate, nr_runs = read_hist_from_data_dir(args.data_dir, nr_bins)
        pickle_dir = {"freqs":freqs, "spec_hists":spec_hists, "sampling_rate":sampling_rate, "nr_runs":nr_runs}
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(pickle_dir, pickle_file)


    # convert hists to pdf
    normalization_factor = np.sum(spec_hists, axis = -1) * np.diff(bin_edges)[0]
    spec_hists = np.divide(spec_hists, normalization_factor[:, :, np.newaxis])

    pdf = PdfPages(f"figures/spec_hist/spec_hist_s{station_nr}_sigmas_{clean}.pdf")
    for ch in range(24):
        sigmas = []
        sigmas_std = []
        freqs_masked = []
        for freq in freqs:
            freq_idx = np.digitize(freq, freqs)
            print(freq_idx)
            print(len(spec_hists))
            print(spec_hists.shape)
            try:
                param, cov = curve_fit(rayleigh, bin_centers, spec_hists[ch, freq_idx-1], p0 = [0.01])
                sigmas.append(param[0])
                sigmas_std.append(np.sqrt(cov[0, 0]))
                freqs_masked.append(freq)
            except:
                continue
        fig, ax = plt.subplots()
        ax.errorbar(freqs_masked, sigmas, yerr = sigmas_std)
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel(r"$\sigma$")

        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
