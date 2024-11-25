import json
import logging
import os
import sys
import argparse
import numpy as np  
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.time import Time

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import NuRadioReco.modules.channelBandPassFilter

from utility_functions import rms_from_temp

def read_rms_from_pickle(file : str) -> np.ndarray:
    with open(file, "rb") as f:
        rms_list = pickle.load(f)
    return np.array(rms_list)


def reduce_by_db(variable, db):
    db_coeff = 10**(-1*db/20.)
    variable = db_coeff * variable
    return variable


def make_std_text(ax, channel_classes, std_comparison, text_x = 0.02, text_y = 0.845,
                  text_kwargs = dict(fontsize = 14, verticalalignment = "top"),
                  box_kwargs = dict(boxstyle = "round", facecolor = "white", alpha = 0.5)):
    text = f" {channel_classes[0]}: {std_comparison[0]:.2f} %"
    for i, group_name in enumerate(channel_classes[1:]):
        text += "\n" + f" {group_name}: {std_comparison[i]:.2f} %"
    ax.text(text_x, text_y, text, **text_kwargs, bbox = box_kwargs, transform = ax.transAxes)


def plot_rms(rms : np.ndarray, ax, channel_mapping, channel_classes,
             decibel_offset = None,
             skip_channels = [], ls = "-", marker = "o", ax_type = "solo",
             db_offset_for_plot = None,
             colors = ["red", "pink", "blue", "black", "green", "yellow", "purple"]):

    print(rms.shape)
    channels = np.arange(24)
    rms_means = np.mean(rms, axis = -1) * units.V
    rms_means = np.squeeze(rms_means)
    rms_std = np.std(rms, axis = -1)
    rms_std = np.squeeze(rms_std)

    if decibel_offset is not None:
        db_coeff = 10**(-1*decibel_offset/20.)
        rms_means = db_coeff * rms_means
        # multiplying by a coefficent is propagated to the uncertainties as the same multiplication
        rms_std = db_coeff * rms_std
     
    first_idx = 0
    for i, (ch_idx, ch_class) in enumerate(zip(channel_mapping, channel_classes)):
        for skip in skip_channels:
            if skip in ch_idx:
                ch_idx.remove(skip)

        last_idx = first_idx + len(rms_means[ch_idx])
        label = f"{ch_class}"
        ax.errorbar(np.arange(first_idx, last_idx), rms_means[ch_idx] / units.mV, yerr = rms_std[ch_idx] / units.mV, label = label, marker = marker, ls = ls, color = colors[i])
        first_idx = last_idx

    ax.set_xlabel("channels", size = "xx-large")
    if ax_type == "solo":
        appendix = ""
    elif ax_type == "first":
        appendix = " (before det removal)" 
    elif ax_type == "second":
        appendix = " (after det removal)"

    ax.set_ylabel("mean rms per channel / mV" + appendix, size = "xx-large")
    if ax_type in ["solo", "first"]:
        legend_antenna = ax.legend(loc = 4, fontsize = "x-large")
    else:
        lines = [Line2D([0], [0], color = "black", ls = "-", marker = "o", lw = 4, ms = 10),
                 Line2D([0], [0], color = "black", ls = "--", marker = "v", lw = 4, ms = 10)]
        legend_antenna = ax.get_legend()
        legend_det = ax.legend(lines, [f"with detector (offset by {db_offset_for_plot} dB)", "without detector"],
                               loc = "best", fontsize = "x-large")
        ax.add_artist(legend_det)
        ax.add_artist(legend_antenna)
    ax.grid()
    channel_mapping_flat = [c for ch_list in channel_mapping for c in ch_list]
    if len(skip_channels) == 0:
        ch_ticks = channels
    else:
        ch_ticks = channels[0:-1*len(skip_channels)]
    ax.set_xticks(ch_ticks, channel_mapping_flat)
    return ch_ticks



def plot_on_double(rms_list, args, channel_mapping, channel_classes, std_comparison = None, figsize = (18, 8), plotting_function = plot_rms):
    if len(args.rms_files) == 1 and args.subplots:
        print("Cannot plot a single rms set on two subplots")
        sys.exit()

    fig, axs = plt.subplots(1, 2, figsize = figsize)
    plotting_function(rms_list[0], axs[0], channel_mapping, channel_classes, skip_channels = args.skip_channels, ax_type = "solo")
    axs[0].set_title("before detector removal", fontsize = "xx-large")
    plotting_function(rms_list[1], axs[1], channel_mapping, channel_classes, skip_channels = args.skip_channels, ax_type = "solo")
    axs[1].set_title("after detector removal", fontsize = "xx-large")
    if std_comparison is not None and channel_classes is not None:
        make_std_text(axs[1], channel_classes=channel_classes, std_comparison=std_comparison)
    for ax in axs:
        ax.tick_params(axis = "both", labelsize = "x-large")

    station_nr = args.rms_files[1].split("_s", 1)[1][0:2]
    fig.suptitle(f"station {station_nr}", fontsize = "xx-large")
    fig.tight_layout()

    filename = args.rms_files[1].split("/")[-1].split(".")[0] + "_comparison"

    fig_file = f"figures/plot_{filename}"
    print(f"Saving as {fig_file}")
    fig.savefig(fig_file, bbox_inches = "tight")

    return



def plot_on_single(rms_list, args, channel_mapping, channel_classes,
                   detector,
                   plotting_function = plot_rms, std_comparison = [],
                   decibel_offset = 55,
                   figsize = (12, 8),
                   thermal_noise_with_detector = False):


    fig, ax = plt.subplots(figsize = figsize)
    ch_ticks =plotting_function(rms_list[0], ax, channel_mapping, channel_classes, skip_channels = args.skip_channels, ax_type = "first",
                                decibel_offset = decibel_offset)
    ax.grid()
    ax.tick_params(axis = "both", labelsize = "x-large")

    if len(rms_list) == 2:
        plotting_function(rms_list[1], ax, channel_mapping, channel_classes,
                          skip_channels=args.skip_channels, ls = "--", marker = "v", ax_type = "second",
                          db_offset_for_plot=decibel_offset)
        ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())))

        #make_std_text(ax, channel_classes, std_comparison)
        station_nr = args.rms_files[1].split("/")[-1].split("station")[-1][0:2]
    else:
        station_nr = args.rms_files[0].split("/")[-1].split("station")[-1][0:2]

    # including thermal noise temps
    noise_temperatures = np.arange(200, 900, 80)
    if thermal_noise_with_detector:
        cst_Vrms_list = [[rms_from_temp(noise_temp_channel, detector, station=int(station_nr), channel=channel, include_amps=True)
                          for channel in range(24)] for noise_temp_channel in noise_temperatures]
        cst_Vrms_list = np.array(cst_Vrms_list)
        cst_Vrms_list = reduce_by_db(cst_Vrms_list, db=decibel_offset)
    else:
        cst_Vrms_list = [[rms_from_temp(noise_temp_channel, detector, station=int(station_nr), channel=channel, include_amps=False)
                          for channel in range(24)] for noise_temp_channel in noise_temperatures]
        cst_Vrms_list = np.array(cst_Vrms_list)

    cmap = mpl.colormaps['inferno']
    temp_colors = cmap(np.linspace(0, 1, len(cst_Vrms_list)))
    temp_plots = []
    temp_labels = []
    for idx, Vrms in enumerate(cst_Vrms_list):
        print(ch_ticks)
        print(Vrms / units.mV)
        print(f"{noise_temperatures[idx]}")
        if idx == 0:
            t_plot = ax.fill_between(ch_ticks,
                                     Vrms[ch_ticks] / units.mV, color = temp_colors[idx], zorder = -1*idx, alpha = 0.7)
        else:
            t_plot = ax.fill_between(ch_ticks, cst_Vrms_list[idx - 1, ch_ticks] / units.mV,
                                     Vrms[ch_ticks] / units.mV, color = temp_colors[idx], zorder = -1*idx, alpha = 0.7)
        temp_plots.append(t_plot)
        temp_labels.append(f"{noise_temperatures[idx]} K")

    ax.add_artist(plt.legend(temp_plots, temp_labels, loc = "upper left")) 

    fig.suptitle(f"station {station_nr}", fontsize = "xx-large")
    fig.tight_layout()

    if len(rms_list) == 1:
        filename = args.rms_files[0].split("/")[-1].split(".")[0]
    else:
        filename = args.rms_files[1].split("/")[-1].split(".")[0] + "_comparison"


    fig_file = f"figures/rms/station{station_nr}/plot_{filename}"
    print(f"Saving as {fig_file}")
    fig.savefig(fig_file, bbox_inches = "tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "plot mean rms per channel")
    parser.add_argument("-d", "--rms_files",
                        nargs = "+",
                        default = ["/user/rcamphyn/noise_study/rms_lists/rms_s24_clean.pickle"],
                        help = "rms pickle file to plot, can choose to plot 2. First one is labeled as before cleaning \
                                second one as after cleaning")
    parser.add_argument("--skip_channels", type = int, nargs = "+", default = [],
                        help = "Choose channels to skip when plotting \
                                e.g. --skip_channels 1 will plot the rms of all channels except 1")
    parser.add_argument("--subplots", action="store_true",
                        help = "Option to decide whether to plot 2 rms lists on a single plot or\
                                two subplots")
    args = parser.parse_args()

    if len(args.rms_files) not in [1, 2]:
        print("Can only plot one or two rms lists, more do not fit on axes.\
              This function is meant to plot before and afters or singles")
        sys.exit()

    if len(args.rms_files) == 2:
        station_nr = args.rms_files[1].split("/")[-1].split("station")[-1][0:2]
    else:
        station_nr = args.rms_files[0].split("/")[-1].split("station")[-1][0:2]

    config_path = f"{os.path.dirname(args.rms_files[0])}/config_s{station_nr}.json"
    print(config_path)

    with open(config_path, "r") as config_json:
        config = json.load(config_json)

    
    detector = detector.Detector(source="rnog_mongo",
                                 always_query_entire_description=False,
                                 database_connection="RNOG_public",
                                 select_stations=int(station_nr),
                                 log_level=logging.INFO)
    
    detector.update(Time(config["detector_time"]))



    rms_list = [read_rms_from_pickle(rms_file) for rms_file in args.rms_files]


    # channel_classes = ["PA", "VPOL PS", "HPOL PS", "VPOL H1", "HPOL H1", "VPOL H2", "HPOL H2", "Surface"]
    # channel_mapping = [[0, 1, 2, 3], [5, 6, 7], [4, 8], [9, 10], [11], [22, 23], [21], [12, 13, 14, 15, 16, 17, 18, 19, 20]]
    channel_mapping = [[0, 1, 2, 3, 9, 10, 22, 23], [5, 6, 7], [4, 8, 11, 21], [12, 14, 15, 17, 18, 20], [13, 16, 19]]
    channel_classes = ["low Vpols", "upper Vpols", "Hpols", "downward LPDA", "upward LPDA"]

    std_comparison = []
    if len(rms_list) == 1:
        ax_type = "solo"
    else:
        ax_type = "first"
#        rms_std_per_group = np.array([[np.std(rms[0, ch_idx] / units.mV) for ch_idx in channel_mapping] for rms in rms_list]) # in mV
#        std_comparison = (rms_std_per_group[1])/(rms_std_per_group[0]) * 100 # in percentage

    if args.subplots:
        plot_on_double(rms_list, args, channel_mapping=channel_mapping, channel_classes=channel_classes, std_comparison=std_comparison)
    else:
        plot_on_single(rms_list, args, channel_mapping=channel_mapping, channel_classes=channel_classes,
                       detector=detector, std_comparison=std_comparison)
