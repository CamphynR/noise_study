import sys
import argparse
import numpy as np  
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from NuRadioReco.utilities import units

def read_rms_from_pickle(file : str) -> np.ndarray:
    with open(file, "rb") as f:
        rms_list = pickle.load(f)
    return np.array(rms_list)

def plot_rms(rms : np.ndarray, ax, skip_channels = [], ls = "-", marker = "o", ax_type = "solo"):
    channel_classes = ["PA", "VPOL PS", "HPOL PS", "VPOL H1", "HPOL H1", "VPOL H2", "HPOL H2", "Surface"]
    channel_mapping = [[0, 1, 2, 3], [5, 6, 7], [4, 8], [9, 10], [11], [22, 23], [21], [12, 13, 14, 15, 16, 17, 18, 19, 20]]


    channels = np.arange(24)
    rms_means = np.mean(rms, axis = -1) * units.V
    rms_std = np.std(rms, axis = -1)
    for ch_idx, ch_class in zip(channel_mapping, channel_classes):
        for skip in skip_channels:
            if skip in ch_idx:
                ch_idx.remove(skip)
        
        ax.errorbar(channels[ch_idx], rms_means[0, ch_idx] / units.mV, yerr = rms_std[0, ch_idx] / units.mV, label = ch_class, marker = marker, ls = ls)
    ax.set_xlabel("channels", size = "xx-large")
    if ax_type == "solo":
        appendix = ""
    elif ax_type == "first":
        appendix = " (before clean)" 
    elif ax_type == "second":
        appendix = " (after clean)"

    ax.set_ylabel("mean rms per channel / mV" + appendix, size = "xx-large")
    if ax_type in ["solo", "first"]:
        ax.legend(loc = "best", fontsize = "x-large")
    else:
        lines = [Line2D([0], [0], color = "black", ls = "-", marker = "o", lw = 4, ms = 10),
                 Line2D([0], [0], color = "black", ls = "--", marker = "v", lw = 4, ms = 10)]
        ax.legend(lines, ["before cleaning", "after cleaning"], loc = "best", fontsize = "x-large")
    ax.grid()
    ax.set_xticks(channels, channels)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "plot mean rms per channel")
    parser.add_argument("-d", "--rms_files",
                        nargs = "+",
                        default = ["/user/rcamphyn/noise_study/rms_lists/rms_s24_clean.pickle"],
                        help = "rms pickle file to plot, can choose to plot 2. First one is labeled as before cleaning \
                                second one as after cleaning")
    parser.add_argument("--skip_channels", type = int, nargs = "+", default = [])
    args = parser.parse_args()

    if len(args.rms_files) not in [1, 2]:
        print("Can only plot one or two rms lists, more do not fit on axes.\
              This function is meant to plot before and afters")
        sys.exit()

    rms_list = [read_rms_from_pickle(rms_file) for rms_file in args.rms_files]
    
    if len(rms_list) == 1:
        ax_type = "solo"
    else:
        ax_type = "first"

    fig, ax = plt.subplots(figsize = (12, 8))
    plot_rms(rms_list[0], ax, skip_channels = args.skip_channels, ax_type = ax_type)
    ax.tick_params(axis = "both", labelsize = "x-large")

    if len(rms_list) == 2:
        ax2 = ax.twinx()
        plot_rms(rms_list[1], ax2, skip_channels=args.skip_channels, ls = "--", marker = "v", ax_type = "second")
        ax2.tick_params(axis = "both", labelsize = "x-large")

    fig.suptitle("station 24", fontsize = "xx-large")
    fig.tight_layout() 
    

    if len(rms_list) == 1:
        filename = args.rms_files[0].split("/")[-1].split(".")[0]
    else:
        filename = args.rms_files[1].split("/")[-1].split(".")[0] + "_comparison"


    fig_file = f"figures/plot_{filename}"
    print(f"Saving as {fig_file}")
    fig.savefig(fig_file, bbox_inches = "tight")