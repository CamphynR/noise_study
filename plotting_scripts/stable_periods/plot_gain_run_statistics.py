import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
import pickle

from NuRadioReco.utilities import units
from rnog_data.runtable import RunTable

from utilities.utility_functions import read_pickle



def per_run_metric(gains, nr_stable_gains, gain_idx, channel_id):
    metric = np.abs(gains[gain_idx, channel_id] - gains[gain_idx - 1, channel_id])/gains[gain_idx - 1, channel_id]
    return metric


def compare_means(gains, nr_stable_gains, gain_idx, channel_id):
    gains_mean_left = np.mean(gains[gain_idx - nr_stable_gains : gain_idx, channel_id])
    gains_mean_right = np.mean(gains[gain_idx + 1 : gain_idx + nr_stable_gains, channel_id])
    metric = np.abs(gains_mean_left - gains_mean_right) / gains_mean_left
    return metric


def compare_to_mean(gains, nr_stable_gains, gain_idx, channel_id):
    gains_mean_left = np.mean(gains[gain_idx - nr_stable_gains : gain_idx, channel_id])
    metric = np.abs(gains[gain_idx, channel_id] - gains_mean_left) / gains_mean_left
    return metric

def sliding_mean(gains, nr_stable_gains, gain_idx, channel_id):
    gains_window = gains[gain_idx: gain_idx+nr_stable_gains, channel_id]
    return np.mean(gains_window)


    
def construct_gains_sublist(gains, seasons_sub, station_ids_sub, channel_ids_sub, exclude=None):
    gains_sub = []
    for season in seasons_sub:
        for station_id in station_ids_sub:
            for channel_id in channel_ids_sub:
                if exclude is None:
                    pass
                elif channel_id in exclude[str(season)][str(station_id)]:
                    continue
                gains_sub.append(gains[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)])
    return np.array(gains_sub)


def construct_dgains_sublist(gains, seasons_sub, station_ids_sub, channel_ids_sub, exclude=None):
    dgains_sub = []
    for season in seasons_sub:
        for station_id in station_ids_sub:
            for channel_id in channel_ids_sub:
                if exclude is None:
                    pass
                elif channel_id in exclude[str(season)][str(station_id)]:
                    continue
                dg_ch = np.diff(gains[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)])
                dgains_sub.append(dg_ch)
    return dgains_sub






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_appendix", default=None)
    args = parser.parse_args()
    seasons = [2023]
    season_ints = []
    for season in seasons:
        if season == "2024_radiant_v2":
            season_ints.append(2024)
        else:
            season_ints.append(season)
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    channel_ids = list(np.arange(24))

    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    known_broken_channels[str(2023)][str(12)] = np.arange(24)
    known_broken_channels[str(2023)][str(22)] = np.arange(24)
    
    known_broken_channels["2024_radiant_v2"][str(12)] = np.arange(24)
    known_broken_channels["2024_radiant_v2"][str(21)] = np.arange(24)
    known_broken_channels["2024_radiant_v2"][str(22)] = np.arange(24)


    


    # rt = RunTable()
    # runtable_kwargs = dict(
    #         stations=station_ids,
    #         start_time=f"{season_ints[0]}-01-01",
    #         stop_time=f"{season_ints[-1]}-12-31",
    #         run_types=["physics"]
    #         )
    # table = rt.get_table(**runtable_kwargs)

    # forced_trigger_idx = table["trigger_soft_enabled"] == 1
    # table = table[forced_trigger_idx]






    times = []
    gains = []
    dgains = [[] for channel_id in channel_ids]
    dgains_min = [[] for channel_id in channel_ids]
    dgains_all = []
    dgains_all_min = []
    for season in seasons:
        for station_id in station_ids:
            # # seasonal calibration used for these runs per season
            # calibration_season_path = f"absolute_amplitude_results/season{seasons[0]}/station{station_id}/default/absolute_amplitude_calibration_season{seasons[0]}_st{station_id}_best_fit.csv"
            # calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            # gain_season = calibration_season["gain"]

            if season == "2024_radiant_v2" and station_id in [12, 21, 22]:
                continue
            cal_per_run_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/slope_fixed_to_2023/season{season}_st{station_id}_all_runs_compiled_slope_fixed_to_2023.pickle"

            with open(cal_per_run_path, "rb") as file:
                cal_per_run = pickle.load(file)

            # table_season = table[table["run"].isin(cal_per_run["run_nr"])]
            # times.extend(table_season["time_start"])


            run_numbers = np.array(cal_per_run["run_nr"])

            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            for channel_id in channel_ids:
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue
                dg_right = 100* np.abs(np.diff(gains_per_run.T[channel_ids.index(channel_id)])) / gains_per_run.T[channel_ids.index(channel_id)][:-1]
                dg_left = 100* np.abs(np.diff(gains_per_run.T[channel_ids.index(channel_id)])) / gains_per_run.T[channel_ids.index(channel_id)][1:]
                dg_min = np.min([dg_right[1:], dg_left[:-1]],
                                axis=0)
                    

                dgains[channel_ids.index(channel_id)].extend(dg_right)
                dgains_min[channel_ids.index(channel_id)].extend(dg_min)
                dgains_all.extend(dg_right)
                dgains_all_min.extend(dg_min)


    times = np.array(times)




    plt.style.use("astroparticle_physics") 
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figname = f"figures/gain_per_run_hist" 

    if len(station_ids) == 1:
        figname += f"_st{station_id}"
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)



    fig, ax = plt.subplots()
        
    hist, bins, patches = ax.hist(dgains_all,
                histtype="stepfilled",
                facecolor=colors[0] + "10",
                edgecolor=colors[0],
                label="one side",
                lw=3.)

    ax.hist(dgains_all_min,
    # bins=bins,
                histtype="stepfilled",
                facecolor=colors[1] + "10",
                edgecolor=colors[1],
                label="minimum of neighbours",
                lw=3.)

    ax.set_yscale("log")
    ax.set_xlabel("diff(gain per run) / %")
    ax.legend()
    fig.suptitle(f"all channels")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


    for channel_id in channel_ids:
        fig, ax = plt.subplots()
            
        hist, bins, patches = ax.hist(dgains[channel_id],
                    histtype="stepfilled",
                    facecolor=colors[0] + "10",
                    edgecolor=colors[0],
                    lw=3.)

        ax.hist(dgains_min[channel_id],
                # bins=bins,
                    histtype="stepfilled",
                    facecolor=colors[1] + "10",
                    edgecolor=colors[1],
                    label="min of neighbours",
                    lw=3.)

        ax.set_yscale("log")
        ax.set_xlabel("diff(gain per run) / %")
        ax.legend()
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)


    # exclude = known_broken_channels
    # gains_all = construct_gains_sublist(gains, seasons, station_ids, channel_ids, exclude=exclude)



    pdf.close()