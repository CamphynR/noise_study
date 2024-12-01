import argparse
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

from NuRadioReco.utilities import units
from utility_functions import open_pickle

def find_config(data_dir):
    """
    function that yields config file, assuming the config to be stored on the same level as the stationX folder.
    """
    job_folder = Path(data_dir).parents[1]
    return str(job_folder) + "/config.json"

def read_hist_from_data_dir(data_dir):
    hist_output_list = []
    run_files = glob.glob(f"{data_dir}/*.pickle")
    for run_file in run_files:
        hist_output = open_pickle(run_file)
        hist_output = hist_output["var"]
        # output shape is [events, channels, 3], where 2 is freqs, bin_idxs or sampling_rate
        hist_output_list += hist_output

    freqs = [[hist_ch[0] for hist_ch in hist_eventlist] for hist_eventlist in hist_output_list]
    freqs = np.array(freqs[0])

    bin_idxs = [[hist_ch[1] for hist_ch in hist_eventlist] for hist_eventlist in hist_output_list]
    bin_idxs = np.array(bin_idxs)
    bin_idxs = np.moveaxis(bin_idxs, 0, 2)
    bins = np.linspace(0, 128, 128)
    spec_hists = np.array([[np.histogram(bin_idxs_freq, bins)[0] for bin_idxs_freq in bin_idxs_ch] for bin_idxs_ch in bin_idxs])
    # spec_hists shape is (channels, frequencies, hists)
    
    sampling_rate = [[hist_ch[2] for hist_ch in hist_eventlist] for hist_eventlist in hist_output_list]
    sampling_rate = np.array(sampling_rate)
    
    if not np.all([freqs[i] == freqs[i - 1] for i in range(1, len(freqs))]):
        raise ValueError("Not every run spectrogram uses the same freqs,\
                make sure the runs contain correct traces and sampling rates")
    elif not np.all([sampling_rate[i] == sampling_rate[i - 1] for i in range(1, len(sampling_rate))]):
        raise ValueError("Not every run spectrogram uses the sampling rate,\
                make sure the runs contain correct traces and sampling rates")

    return freqs, spec_hists, sampling_rate


def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 *sigma**2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to plot spec histograms")
    parser.add_argument("--data_dir", "-dir", nargs = "+", help = "directory containing the run pickles")
    parser.add_argument("--channel", "-ch", type=int, nargs = "+", default=[0])
    parser.add_argument("--freq", type=float, default=400, help="frequency in MHz")
    args = parser.parse_args()
    if args.channel is None:
        args.channel = list(range(24))
    freq = args.freq * units.MHz

    config_path = find_config(args.data_dir[0])
    with open(config_path) as config_file:
        config = json.load(config_file)
    hist_range = config["variable_function_kwargs"]["clean"]["range"]
    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
    bins = np.linspace(hist_range[0], hist_range[1], nr_bins)
    bin_centers = bins[:-1] + np.diff(bins) / 2


    freqs, spec_hists, sampling_rate = read_hist_from_data_dir(args.data_dir[0])
    print(freqs[args.channel].shape)
    freq_idx = np.digitize(freq, freqs[args.channel[0]])

    # convert hists to pdf
    normalization_factor = np.sum(spec_hists, axis = -1) * np.diff(bins)[0]
    spec_hists = np.divide(spec_hists, normalization_factor[:, :, np.newaxis])

    # Rayleigh spectra fits


    pdf = PdfPages("test.pdf")
    for ch in args.channel:
        param, cov = curve_fit(rayleigh, bin_centers, spec_hists[ch, freq_idx-1])
        cov = np.sqrt(cov)[0, 0]
        ray = rayleigh(bin_centers, sigma=param[0])
        fig, ax = plt.subplots()
        ax.stairs(spec_hists[ch, freq_idx - 1], edges = bins)
        ax.plot(bin_centers, ray)
        ax.text(0.4, 4, r"$\sigma=$" + f"{param[0]:.2f}" + r"$\pm$" +  f"{cov:.2f}")
        ax.set_xlabel("Spectrum magnitude / V/GHz", size = "large")
        ax.set_ylabel("N")
        ax.set_title(f"Channel {ch}, freq = {freqs[ch, freq_idx - 1]} GHz")
        fig.savefig(pdf, format="pdf")
    pdf.close()
