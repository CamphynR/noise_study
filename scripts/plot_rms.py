import os
import argparse
import numpy as np  
import pickle
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units

def read_rms_from_pickle(file : str) -> np.ndarray:
    with open(file, "rb") as f:
        rms_list = pickle.load(f)
    return np.array(rms_list)

def plot_rms(rms : np.ndarray, fig_file : str, figsize = (12, 8)):
    channel_classes = ["PA", "VPOL PS", "HPOL PS", "VPOL H1", "HPOL H1", "VPOL H2", "HPOL H2", "Surface"]
    channel_mapping = [[0, 1, 2, 3], [5, 6, 7], [4, 8], [9, 10], [11], [22, 23], [21], [12, 13, 14, 15, 16, 17, 18, 19, 20]]

    fig, ax = plt.subplots(figsize = figsize)
    channels = np.arange(24)
    rms_means = np.mean(rms, axis = 1) * units.V
    rms_std = np.std(rms, axis = 1)
    for ch_idx, ch_class in zip(channel_mapping, channel_classes):
        ax.errorbar(channels[ch_idx], rms_means[ch_idx] / units.mV, yerr = rms_std[ch_idx] / units.mV, label = ch_class, marker = "o")
    ax.set_xlabel("channels", size = "xx-large")
    ax.set_ylabel("mean rms per channel / mV", size = "xx-large")
    ax.legend(loc = "best")
    ax.grid()
    fig.suptitle("station 24")
    fig.tight_layout()
    fig.savefig(fig_file, bbox_inches = "tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "plot mean rms per channel")
    parser.add_argument("-f", "--rms_file",
                        default = "/user/rcamphyn/noise_study/rms_lists/rms_s24_clean.pickle",
                        help = "rms pickle file to plot")
    args = parser.parse_args()

    rms = read_rms_from_pickle(args.rms_file)
    fig_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../figures")
    fig_file = f"{fig_dir}/rms_plot"
    plot_rms(rms, fig_file)