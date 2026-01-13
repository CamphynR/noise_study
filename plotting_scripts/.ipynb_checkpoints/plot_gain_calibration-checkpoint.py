import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from modules.plotting_functions import convert_to_db




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    gain_directory = "absolute_amplitude_results"
    seasons = []
    stations = []
    frames = []
    for gain_file in os.listdir(gain_directory):
        season = gain_file.split("season")[1][0:4]
        seasons.append(season)
        station_id = gain_file.split("st")[1][0:2]
        stations.append(station_id)

        ds = pd.read_csv(gain_directory + "/" + gain_file)
        ds = ds.apply(convert_to_db)
        ds.rename({0 : station_id}, inplace=True)
        frames.append(ds)

    index=[np.array(seasons), np.array(stations)]
    ds = pd.concat(frames, keys=seasons)

    print(ds.loc['2023'])

    figdir = "figures/calibrated_gain_summaries"

    plt.style.use("retro")

    station_ids = ds.loc['2022'].index.tolist()

    channel_grouping = [0, 1, 2, 3, 4, 8, ]

    for station_id in station_ids:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].scatter(np.arange(24), ds.loc['2023', station_id], label = "season 2023")
        axs[0].scatter(np.arange(24), ds.loc['2022', station_id], label = "season 2022")
        axs[0].set_ylabel("Gain / dB")
        axs[0].legend()
        axs[1].plot(ds.loc['2023', station_id] - ds.loc['2022', station_id])
        axs[1].set_ylabel("Gain diff / dB")
        axs[1].set_xlabel("channels")
        plt.savefig(figdir + "/" + f"calibrated_gains_st{station_id}.png")
