import argparse
import glob
import json
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare


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


def chi2(observed, expected):
    chi2 = np.square(observed-expected)
    chi2 = chi2/expected
    chi2 = np.sum(chi2)
    dof = len(observed) - 1
    print(dof)
    return chi2/dof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-dir", help="Directory containing run pickles (should be named clean or raw)")
    parser.add_argument("--freq_lim", type=float, nargs=2, default=[0.2, 0.6])
    args = parser.parse_args()
    args.channel = 0
    station_nr = int(str(Path(args.data_dir).parents[0]).split("/")[-1][-2:])
    clean = args.data_dir.split("/")[-1]

    config_path = find_config(args.data_dir)
    with open(config_path) as config_file:
        config = json.load(config_file)

    hist_range = config["variable_function_kwargs"][f"{clean}"]["range"]
    nr_bins = config["variable_function_kwargs"][f"{clean}"]["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    pickle_path = f"data/plotting_data/spec_hist/spec_hist_s{station_nr}_{clean}_no_cw.pickle"
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


    nr_of_freqs = 10
    # the detector response becomes almost 0 after 1000 MHz
    freq_idxs = np.nonzero((args.freq_lim[0] < freqs) & (freqs < args.freq_lim[1]))[0]
    steps = int(len(freq_idxs)/nr_of_freqs)
    freq_idxs = freq_idxs[::steps]

    # convert hists to pdf
    normalization_factor = np.sum(spec_hists, axis = -1) * np.diff(bin_edges)[0]
    spec_hists = np.divide(spec_hists, normalization_factor[:, :, np.newaxis])

    print(len(freq_idxs))

    plt.style.use("/user/rcamphyn/envs/gaudi.mplstyle")
    fig, axs = plt.subplots(len(freq_idxs), 1, sharex=True,
                            figsize = (12, 16))
    fig.suptitle(f"Distribution of spectral magnitude per frequency, station {station_nr}, channel {args.channel}, {clean}")
    for i, ax in enumerate(axs.flatten()):
        freq_idx = freq_idxs[i]
        param, cov = curve_fit(rayleigh, bin_centers, spec_hists[args.channel, freq_idx], p0 = [0.01])
        cov = np.sqrt(cov)[0, 0]
        ray = rayleigh(bin_centers, sigma=param[0])
#        chi2dof = chi2(spec_hists[args.channel, freq_idx,:30], ray[:30])
#        chi2dof = chisquare(spec_hists[args.channel, freq_idx], ray)
        ax.stairs(spec_hists[args.channel, freq_idx], edges=bin_edges)
        ax.plot(bin_centers, ray)
#        ax.text(0.6, 0.3, r"$\chi²/dof$ = " + f"{chi2dof}",
#                transform=ax.transAxes)
        ax.set_title(f"freq = {freqs[freq_idx]:.2f} GHz", x = 0.9, y = 0.5)

    axs[-1].set_xlabel("Spectrum magnitude / V/GHz")
    fig.text(0.04, 0.5, "Counts", va="center", rotation="vertical")

    fig.savefig(f"figures/spec_hist/spec_hist_s{station_nr}_{clean}", dpi=200)
