"""
Plot of general histogram combining all traces that where parsed over.
The original structure is a histogram per run, yielding X amounts of histograms.
The number of bins is chosen the same for every trace, since every trace has a range of [0, 2048], the bin edges should be the same
The individual histograms are combined into one big amplitude histogram.
Note that this histogram contains all frequencies in the bandpass e.g. 200 - 600 MHz
For a histogram over only one frequency see the other histogram plot
"""

import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit

from NuRadioReco.utilities import units
from utility_functions import open_pickle


def read_hist_from_data_dir(data_dir):
    hist_output_list = []
    for run_file in glob.glob(f"{data_dir}/*.pickle"):
        hist_output = open_pickle(run_file)
        hist_output = hist_output["var"]
        # output shape is [events, channels, 2], where 2 is hist or edges
        hist_output_list += hist_output

    hist = [[hist_ch[0] for hist_ch in hist_eventlist] for hist_eventlist in hist_output_list]
    hist = np.array(hist)
    hist_sum = np.sum(hist, axis = 0)
    edges = [[hist_ch[1] for hist_ch in hist_eventlist] for hist_eventlist in hist_output_list]
    edges = np.array(edges)
    
    if not np.all([edges[i] == edges[i - 1] for i in range(1, len(edges))]):
        raise ValueError("Not every run histogram uses the same bin edges,\
                make sure to run the parser with the same bin edges for every trace")
    else:
        edges = edges[0]
        edges = edges / units.mV

    # convert to pdf
#    hist_sum = hist_sum.astype(np.float32)
#    hist_sum[args.channel] = hist_sum[args.channel] / (np.sum(hist_sum[args.channel]) * np.diff(edges[args.channel])[0])

    return hist_sum, edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to plot trace histograms")
    parser.add_argument("--data_dir", "-dir", nargs = "+", help = "directory containing the run pickles")
    parser.add_argument("--channel", "-ch", type=int, nargs = "+", default=None)
    args = parser.parse_args()
    if args.channel is None:
        args.channel = list(range(24))

    if len(args.data_dir) == 2:
        hist_sum_raw, edges_raw = read_hist_from_data_dir(args.data_dir[0])
        hist_sum, edges = read_hist_from_data_dir(args.data_dir[1])
    else: 
        hist_sum, edges = read_hist_from_data_dir(args.data_dir[0])


    def para(x, a, b):
        return -a*x**2 + b

    pdf = PdfPages("figures/amplitudes/amplitude_distributions_log.pdf")
    
    for ch in args.channel:
        centers = edges[ch, :-1] + np.diff(edges[ch])[0]/2
        param, cov = curve_fit(para,
                               centers[np.where(hist_sum[ch]!=0)],
                               np.log10(hist_sum[ch, hist_sum[ch]!=0]))
        fig, ax = plt.subplots()
 
        if len(args.data_dir) == 2:
            ax.stairs(hist_sum_raw[ch], edges_raw[ch], color = "red", label="raw")
        ax.stairs(hist_sum[ch], edges[ch], label="clean")
        ax.plot(centers, 10**para(centers, param[0], param[1]))

        ax.legend(loc = "best")
        ax.set_yscale("log")
        ax.set_xlabel("Amplitude / mV", size = "large")
        ax.set_ylabel("N", size = "large")
        ax.set_title(f"Distribution of noise amplitudes, channel {ch}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")

    pdf.close()
