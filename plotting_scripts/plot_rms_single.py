"""
Script to plot a single rms array
"""

import os
import glob
import json
import logging
import argparse
import numpy as np  
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from NuRadioReco.utilities import units



def read_config(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config



def read_rms_from_pickle(file : str) -> np.ndarray:
    with open(file, "rb") as f:
        rms_list = pickle.load(f)
    return np.array(rms_list)



def make_std_text(ax, channel_classes, std_comparison, text_x = 0.02, text_y = 0.845,
                  text_kwargs = dict(fontsize = 14, verticalalignment = "top"),
                  box_kwargs = dict(boxstyle = "round", facecolor = "white", alpha = 0.5)):
    text = f" {channel_classes[0]}: {std_comparison[0]:.2f} %"
    for i, group_name in enumerate(channel_classes[1:]):
        text += "\n" + f" {group_name}: {std_comparison[i]:.2f} %"
    ax.text(text_x, text_y, text, **text_kwargs, bbox = box_kwargs, transform = ax.transAxes)



def plot_rms(rms : np.ndarray, ax, channel_mapping, channel_classes, config, skip_channels = [],
             ls = "-", marker = "o"):
    channels = np.arange(24)
    if not config["only_mean"]:
        rms_means = np.mean(rms, axis = -1) * units.V
        rms_std = np.std(rms, axis = -1)

    first_idx = 0
    for ch_idx, ch_class in zip(channel_mapping, channel_classes):
        for skip in skip_channels:
            if skip in ch_idx:
                ch_idx.remove(skip)

        last_idx = first_idx + len(rms_means[ch_idx])
        label = f"{ch_class}"
        ax.errorbar(np.arange(first_idx, last_idx),
                    rms_means[ch_idx] / units.mV,
                    yerr = rms_std[ch_idx] / units.mV,
                    label = label, marker = marker, ls = ls)
        first_idx = last_idx

    ax.set_xlabel("channels", size = "xx-large")
    ax.set_ylabel("mean rms per channel / mV", size = "xx-large")
    ax.legend(loc = 4, fontsize = "x-large")
    ax.grid()
    channel_mapping_flat = [c for ch_list in channel_mapping for c in ch_list]
    if len(skip_channels) == 0:
        ch_ticks = channels
    else:
        ch_ticks = channels[0:-1*len(skip_channels)]
    ax.set_xticks(ch_ticks, channel_mapping_flat)
    ax.tick_params(axis = "both", labelsize = "x-large")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "plot mean rms per channel")
    parser.add_argument("rms_pickle",
                        default = "/user/rcamphyn/noise_study/rms_lists/rms_s24_clean.pickle",
                        help = "rms pickle file to plot")
    parser.add_argument("--skip_channels", type = int, nargs = "+", default = [],
                        help = "Choose channels to skip when plotting \
                                e.g. --skip_channels 1 will plot the rms of all channels except 1")
    parser.add_argument("--config", default = "config.json")
    parser.add_argument("--debug", action = "store_true")
    args = parser.parse_args()
    logger = logging.getLogger("Noise.plot_rms_single.py")
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)
    # look for config settings in same folder as pickle, else take standard config in main folder
    config_path = glob.glob(f"{os.path.dirname(args.rms_pickle)}/config.json")
    if len(config_path) != 0:
        config_run = read_config(config_path[0])
    else:
        config_run = read_config(args.config)

    # shape is (channels, variables)
    rms_list = read_rms_from_pickle(args.rms_pickle)

    channel_mapping = [[0, 1, 2, 3, 9, 10, 22, 23], [5, 6, 7], [4, 8, 11, 21], [12, 14, 15, 17, 18, 20], [13, 16, 19]]
    channel_classes = ["low Vpols", "upper Vpols", "Hpols", "downward LPDA", "upward LPDA"]

    fig, ax = plt.subplots(figsize = (12, 8))
    plot_rms(rms_list, ax, channel_mapping=channel_mapping, channel_classes=channel_classes, config = config_run, skip_channels = args.skip_channels)
    try:
        station_nr = args.rms_pickle.split("/")[-1].split("station")[-1][0:2]
    except:
        station_nr = 0

    fig.suptitle(f"station {station_nr}", fontsize = "xx-large")
    fig.tight_layout()

    figname = "plot_" + args.rms_pickle.split("/")[-1].split(".")[0]
    fig_file = f"figures/rms/station{station_nr}/{figname}"
    print(f"saving as {fig_file}")
    fig.savefig(fig_file, bbox_inches = "tight")
