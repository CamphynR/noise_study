import argparse
from astropy.time import Time
import copy
import glob
import json
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from scipy import constants

from NuRadioReco.utilities import units, fft

from utilities.utility_functions import read_pickle




if __name__ == "__main__":

    season = 2022
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    nr_channels = 24
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    channel_ids = np.arange(24)
#    channel_ids = [5]
    frequencies = fft.freqs(nr_samples, sampling_rate) 


    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


#------------DATA-------------

    times = [0 for _ in station_ids]
    times_date = [0 for _ in station_ids]
    vrms = [0 for _ in station_ids]
    var_vrms = [0 for _ in station_ids]

    for station_id in station_ids:
        times_st = []
        vrms_st = []
        var_vrms_st = []
        data_paths = natsorted(glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.2/season{season}/station{station_id}/clean/average_vrms_run*.pickle"))

        for pickle in data_paths:
            rms_dict = read_pickle(pickle)
            times_st.append(rms_dict["header"]["begin_time"].unix)
            vrms_st.append(rms_dict["vrms"])
            var_vrms_st.append(rms_dict["var_vrms"])
        vrms_st = np.array(vrms_st).T
        var_vrms_st = np.array(var_vrms_st).T

        times_st = np.array(times_st)
        times_date_st = [Time(t, format="unix").strftime("%Y-%B-%d") for t in times_st]

        times[station_ids.index(station_id)] = times_st
        times_date[station_ids.index(station_id)] = times_date_st
        vrms[station_ids.index(station_id)] = vrms_st
        var_vrms[station_ids.index(station_id)] = var_vrms_st



# ------CALIBRATION--------


    calibration = [0 for _ in station_ids]
    for station_id in station_ids:
        calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
        calibration[station_ids.index(station_id)] = pd.read_csv(calibration_path, index_col=0)
    

#----------------PLOTS------------------


    pdf = PdfPages(f"figures/vrms/vrms_distribution_season{season}_all_stations.pdf")
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
#        hist, bin_edges = np.histogram(vrms[channel_id, :] / units.mV,
#                                       bins=50)
#        ax.bar(bin_edges[:-1], hist,
#               facecolor="white",
#               edgecolor="black",
#               linewidth=1.,
#               align="left")
        for station_id in station_ids:
            ax.hist(vrms[station_ids.index(station_id)][channel_id, :] / units.mV, bins=50, histtype="step",
                    label=f"station {station_id}")

        ax.set_xlabel("Vrms / mV")
        ax.set_ylabel("Counts")
        ax.legend()

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()




    channel_types = {
            "VPol PA" : [0, 1, 2, 3],
            "VPol" : [5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA down" : [13, 16, 19],
            "LPDA up" : [12, 14, 15, 17, 19, 20]}

    bins = np.linspace(0.3, 1., 50)

#    for channel_type, channel_ids in channel_types.items():
#        vrms_calibrated = []
#        for channel_id in channel_ids:
#            vrms_calibrated_tmp = vrms[channel_id] / calibration["gain"][channel_id]
#            vrms_calibrated.append(vrms_calibrated_tmp)
#
#
#        fig, ax = plt.subplots()
#        ax.hist(np.ndarray.flatten(vrms[channel_ids]) / np.max(vrms[channel_ids]), bins=50, histtype="step", label="raw")
#        ax.hist(vrms_calibrated / np.max(vrms_calibrated), bins=50, histtype="step", label="calibrated")
#
#        ax.set_xlabel("normalized Vrms")
#        ax.set_ylabel("Counts")
#        ax.legend()
#
#        fig.tight_layout()
#        figname = f"figures/vrms/vrms_distribution_season{season}_st{station_id}_{channel_type}_calibrated.png"
#        fig.savefig(figname, dpi=150)

    
    plt.close()

    # RAW
    # -----------
    fig, ax = plt.subplots()
    for i, (channel_type, channel_ids) in enumerate(channel_types.items()):

        for station_id in station_ids:
            hist, bins, patches = ax.hist(np.ndarray.flatten(vrms[station_ids.index(station_id)][channel_ids]) / np.max(vrms[station_ids.index(station_id)][channel_ids]),
                    bins=bins, histtype="stepfilled", label=channel_type,
                    facecolor=colors[i] + "80",
                    edgecolor=colors[i],
                    lw=3,
                    zorder=-i*100)

    ax.set_xlabel("normalized Vrms / a.u.")
    ax.set_ylabel("# data runs")
    ax.set_xlim(0.3, None)
    ax.legend()

    fig.tight_layout()
    figname = f"figures/vrms/vrms_distribution_season{season}_all_stations_raw.png"
    fig.savefig(figname, dpi=150)





    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))
    axs = np.ndarray.flatten(axs)

    axs[0].hist([0],
            color="black",
            label = "calibrated")
    axs[0].hist([0],
            histtype="step",
            color="black",
            label = "raw")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize="large")
    axs[0].clear()

    for i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        ax = axs[i - i // 4]
        vrms_raw = []
        vrms_calibrated = []


        for station_id in station_ids:
            vrms_raw_st = []
            vrms_calibrated_st = []
            for channel_id in channel_ids:
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue
                vrms_calibrated_tmp = vrms[station_ids.index(station_id)][channel_id] / calibration[station_ids.index(station_id)]["gain"][channel_id]
                vrms_calibrated_st.extend(vrms_calibrated_tmp)

                vrms_raw_st.extend(vrms[station_ids.index(station_id)][channel_id])
            vrms_raw.extend(vrms_raw_st / np.max(vrms_raw_st))
            vrms_calibrated.extend(vrms_calibrated_st / np.max(vrms_calibrated_st))



        hist, bin_edges, patches = ax.hist(vrms_raw,
                bins=bins, histtype="stepfilled",
                facecolor=(0, 0, 0, 0),
                                           lw=3,
                edgecolor=colors[i],
                                           zorder=-i*100)

        ax.hist(vrms_calibrated,
                bins=bins, histtype="stepfilled",
                lw=3,
                facecolor=colors[i] + "cc",
                edgecolor=(0, 0, 0, 0))

        ax.hist([0],
                color=colors[i],
                label=channel_type)


        ax.set_xlim(0.3, None)
        ax.legend(loc="upper left", fontsize="medium")

    for ax_i in [2, 3]:
        axs[ax_i].set_xlabel("normalized Vrms / a.u.")

    for ax_i in [0, 2]:
        axs[ax_i].set_ylabel("# data runs")




    fig.tight_layout()
    figname = f"figures/vrms/vrms_distribution_season{season}_all_stations_calibrated.png"
    fig.savefig(figname, dpi=300, bbox_inches="tight")
