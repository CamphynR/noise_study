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
    station_ids_new_daq = [11, 13, 23]
    station_index_new_daq = [0, 1, 2, 5]
    station_ids_old_daq = [21, 22, 24]
    station_index_old_daq = [3, 4, 6]

    station_ids = station_ids
    
    calibration_type = "measured_noise_no_weight_new_impedance_cable_11"
    
    season = args.season
    channel_type = args.channel_type
    gains = []
    for station_id in station_ids:
        radiant_calibration_path = f"absolute_amplitude_results/season{args.season}/station{station_id}/{calibration_type}/absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_type}_best_fit.csv"
        # backwards compatibility
        if not os.path.exists(radiant_calibration_path):
            radiant_calibration_path = f"absolute_amplitude_results/absolute_amplitude_calibration_season{season}_st{station_id}.csv"
        calibration_parameters = pd.read_csv(radiant_calibration_path)
        gain = calibration_parameters["gain"].to_numpy()
#        el_ampl = calibration_parameters["el_ampl"].to_numpy()
#        el_cst = calibration_parameters["el_cst"].to_numpy()
#        f0 = calibration_parameters["f0"].to_numpy()
        gains.append(gain)

    gains = np.array(gains)
    gains = convert_to_db(gains)
    gains = gains.T


    channel_types = {"VPols" :[0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPols" : [4, 8, 11, 21],
                     "LPDAs" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    plt.style.use("astroparticle_physics")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    markers = ["o", "v", "D", "X", "8", "^", "+", "p", "<", ">", "s"]
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))


    for ax, channel_type in zip(axs, channel_types):
        gains_channel_type = gains[channel_types[channel_type]]
        mean = np.mean(gains_channel_type, axis=0)
        std = np.std(gains_channel_type, axis=0)
        outliers = np.logical_or(np.greater(gains_channel_type, mean + 2*std), np.less(gains_channel_type, mean - 2*std))
        for station_index, station_id in enumerate(station_ids):
            violin = ax.violinplot(gains_channel_type[:, station_index][~outliers[:, station_index]],
                          positions=[station_index + 1], vert=True, showmeans=True)
            # you cannot pass color kwarg to ax.violinplot 
            for part in violin:
                if part == "bodies":
                    for pc in violin[part]:
                        pc.set_color(colors[0])
                else:
                    violin[part].set_color(colors[0])

        for channel_index, channel_id in enumerate(channel_types[channel_type]):
            outliers_channel = outliers[channel_index, :]
            ax.scatter(np.arange(1, len(station_ids)+1)[~outliers_channel],
                       gains.T[~outliers_channel, channel_id],
                       color=colors[0])
            if np.any(outliers_channel == True): 
                ax.scatter(np.arange(1, len(station_ids)+1)[outliers_channel], gains.T[outliers_channel, channel_id],
                           marker=markers[channel_index],
                           label=f"channel {channel_id}",
                           color="gray",
                           s=80)


        ax.legend(loc="lower left", fontsize=21)
        ax.set_xticks(np.arange(1, len(station_ids)+1), station_ids)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.set_xlabel("Station", size=26)
        ax.set_ylabel("Gain / dB", size=26)
        ax.set_title(f"{channel_type}", size=26)
#    fig.suptitle(f"Overview of gain calibration season {season}")
    fig.tight_layout()
    fig.savefig(f"figures/overviews/gain_season{season}.png")
    
