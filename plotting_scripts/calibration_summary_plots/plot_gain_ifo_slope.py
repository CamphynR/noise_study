import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utilities.utility_functions import convert_to_db, convert_error_to_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--channel_type", default="deep", choices=["deep", "hpol", "shallow"])
    args = parser.parse_args()

    station_ids = [11, 12, 13, 21, 23, 24]
#    station_ids = [13, 23]
    station_ids_new_daq = [11, 13, 23]
    station_index_new_daq = [0, 1, 2, 5]
    station_ids_old_daq = [21, 22, 24]
    station_index_old_daq = [3, 4, 6]

    station_ids = station_ids
    
    calibration_type = "default"
    
    season = args.season
    channel_type = args.channel_type
    gains_per_st = [0 for station_id in station_ids]
    gains_error_per_st = [0 for station_id in station_ids]
    gains = []

    slopes_per_st = [0 for station_id in station_ids]
    slopes_error_per_st = [0 for station_id in station_ids]
    slopes = []
    for station_id in station_ids:
        radiant_calibration_path = f"absolute_amplitude_results/season{args.season}/station{station_id}/{calibration_type}/absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_type}_best_fit.csv"
        if calibration_type == "default":
            radiant_calibration_path = f"absolute_amplitude_results/season{args.season}/station{station_id}/{calibration_type}/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
            radiant_calibration_error_path = f"absolute_amplitude_results/season{args.season}/station{station_id}/{calibration_type}/absolute_amplitude_calibration_season{season}_st{station_id}_best_fiterror.csv"
        calibration = pd.read_csv(radiant_calibration_path, index_col=0)
        calibration_error = pd.read_csv(radiant_calibration_error_path, index_col=0)
        gain = calibration["gain"].to_numpy()
        gains_per_st[station_ids.index(station_id)] = gain
        gains_error_per_st[station_ids.index(station_id)] = calibration_error["gain"].to_numpy()
        gains.append(gain)

        slope = calibration["slope"].to_numpy()
        slopes_per_st[station_ids.index(station_id)] = slope
        slopes_error_per_st[station_ids.index(station_id)] = calibration_error["slope"].to_numpy()
        slopes.append(slope)
    
    gains = np.array(gains)
    gains = gains.T

    slopes = np.array(slopes)
    slopes = slopes.T


    channel_types = {"VPols" :[0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPols" : [4, 8, 11, 21],
                     }
#                     "LPDAs" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    plt.style.use("astroparticle_physics")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array(prop_cycle.by_key()["color"])
    colors = colors[[0, 2, 3, 4, 5]]
    markers = ["o", "v", "D", "X", "8", "^", "+", "p", "<", ">", "s"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    handles = []
    labels = []
    for ax_i, (ax, channel_type) in enumerate(zip(axs, channel_types)):
        slopes_channel_type = slopes[channel_types[channel_type]]
        gains_channel_type = gains[channel_types[channel_type]]
        mean = np.mean(gains_channel_type, axis=0)
        std = np.std(gains_channel_type, axis=0)
        outliers = np.logical_or(np.greater(gains_channel_type, mean + 2.2*std), np.less(gains_channel_type, mean - 2.2*std))
        for station_index, station_id in enumerate(station_ids):
            if station_index == 0:
                label=channel_type
            else:
                label=None
            ax.scatter(np.ndarray.flatten(slopes_channel_type[:, station_index][~outliers[:, station_index]]),
                       np.ndarray.flatten(gains_channel_type[:, station_index][~outliers[:, station_index]]),
                       label=label,
                       color=colors[ax_i])

#        for channel_index, channel_id in enumerate(channel_types[channel_type]):
#            label = None
#            if channel_index == 0:
#                label = channel_type
#            outliers_channel = outliers[channel_index, :]
#            ax.scatter(np.arange(1, len(station_ids)+1)[~outliers_channel],
#                       gains.T[~outliers_channel, channel_id],
#                       color=colors[ax_i],
#                       label=label)
#            if np.any(outliers_channel == True): 
#                ax.scatter(np.arange(1, len(station_ids)+1)[outliers_channel], gains.T[outliers_channel, channel_id],
#                           marker=markers[channel_index],
##                           label=f"channel {channel_id}",
#                           color="gray",
#                           alpha=0.5,
#                           s=80)

        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles.extend(handles_ax)
        labels.extend(labels_ax)
        ax.tick_params(axis='both', which='major', labelsize=21)
#        ax.set_xlim(-1, 0)
#        ax.set_title(f"{channel_type}", size=26)
    axs[0].set_ylabel("Gain / amp", size=26)
    fig.legend(handles, labels, ncols=5, loc="lower center", bbox_to_anchor=(0.5, 0.99), fontsize="x-large")
    fig.tight_layout()
    fig.savefig(f"figures/overviews/gain_ifo_slope_season{season}.png", bbox_inches="tight")
    plt.close(fig)


#    channel_ids = np.arange(24)
#    fig, ax = plt.subplots()
#    for station_id in station_ids:
#        ax.errorbar(channel_ids,
#                   convert_to_db(gains_per_st[station_ids.index(station_id)]),
#                    yerr = convert_error_to_db(gains_error_per_st[station_ids.index(station_id)], gains_per_st[station_ids.index(station_id)]),
#                   label=f"station {station_id}",
#                    fmt="o",
#                    ls=None
#                   )
#
#    ax.set_xlabel("channel")
#    ax.set_ylabel("Gain / dB")
#    ax.set_xticks(channel_ids, labels=channel_ids, rotation=-60)
#    ax.legend()
#    ax.set_ylim(53, 67)
#    fig.tight_layout()
#    fig.savefig(f"figures/overviews/gains_season{season}_per_station.png", bbox_inches="tight")
#
#
    
