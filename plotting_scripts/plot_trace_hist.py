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
import numpy as np

from utility_functions import open_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to plot trace histograms")
    parser.add_argument("--data_dir", "-dir", help = "directory containing the run pickles")
    args = parser.parse_args()

    hist_output_list = []

    for run_file in glob.glob(f"{args.data_dir}/*"):
        hist_output = open_pickle(run_file)
        hist_output = hist_output["var"]
        print(len(hist_output[0][0][0]))
        print(len(hist_output[0][0][1]))

        # output shape is [channels, events, 2, hist], where 2 is hist or edges
        hist_output = np.swapaxes(hist_output, 0, 1)
        hist_output_list += hist_output
    
    hist_output_list = np.array(hist_output_list)
    print(hist_output_list.shape)
