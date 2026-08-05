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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_appendix", default=None)
    args = parser.parse_args()
    seasons = [2023]
    station_ids = [11]
    channel_ids = np.arange(24)
    

    rt = RunTable()
    runtable_kwargs = dict(
            stations=station_ids,
            start_time=f"{seasons[0]}-01-01",
            stop_time=f"{seasons[-1]}-12-31",
            run_types=["physics"]
            )
    table = rt.get_table(**runtable_kwargs)

    forced_trigger_idx = table["trigger_soft_enabled"] == 1
    table = table[forced_trigger_idx]

    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    # seasonal calibration used for these runs per season


    channel_types = {
        "VPols" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
        "HPols" : [4, 8, 11, 21],
        "LPDA up" : [13, 16, 19],
        "LPDA down" : [12, 14, 15, 17, 18, 20]}


    times = [[] for _ in station_ids]
    gains_all_slope_fixed = [0 for _ in station_ids]
    gains_all_slope_free = [0 for _ in station_ids]
    gains_all_no_pulser_removed = [0 for _ in station_ids]
    run_nrs = [0 for _ in station_ids]

    default_season_calibration = [0 for _ in station_ids]
    slope_fitted_season_calibration = [0 for _ in station_ids]
    slope_fitted_weight_normalized_season_calibration = [0 for _ in station_ids]

    for season in seasons:
        for station_id in station_ids:

            calibration_season_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
            calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            gain_season = calibration_season["gain"]
            default_season_calibration[station_ids.index(station_id)] = gain_season


            calibration_season_path = f"absolute_amplitude_results/season{season}/station{station_id}/pulser_removed_slope_fitted/absolute_amplitude_calibration_season{season}_st{station_id}_pulser_removed_slope_fitted_best_fit.csv"
            calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            gain_season = calibration_season["gain"]
            slope_fitted_season_calibration[station_ids.index(station_id)] = gain_season


            calibration_season_path = f"absolute_amplitude_results/season{season}/station{station_id}/pulser_removed_slope_fitted_weight_normalized/absolute_amplitude_calibration_season{season}_st{station_id}_pulser_removed_slope_fitted_weight_normalized_best_fit.csv"
            calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            gain_season = calibration_season["gain"]
            slope_fitted_weight_normalized_season_calibration[station_ids.index(station_id)] = gain_season


            cal_per_run_slope_fixed_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/slope_norm_fixed_to_season/season{season}_st{station_id}_all_runs_compiled_slope_norm_fixed_to_season.pickle"
            cal_per_run_slope_free_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/slope_norm_per_run/season{season}_st{station_id}_all_runs_compiled_slope_norm_per_run.pickle"
            cal_per_run_no_pulser_removed = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/season{season}_st{station_id}_all_runs_compiled.pickle"

            with open(cal_per_run_slope_fixed_path, "rb") as file:
                cal_per_run = pickle.load(file)

            table_season = table[table["run"].isin(cal_per_run["run_nr"])]
            table_st = table_season[table_season["station"] == station_id]
            times[station_ids.index(station_id)].extend(table_st["time_start"])

            run_nrs[station_ids.index(station_id)] = cal_per_run["run_nr"]


            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            gains_all_slope_fixed[station_ids.index(station_id)] = gains_per_run


            with open(cal_per_run_slope_free_path, "rb") as file:
                cal_per_run = pickle.load(file)

            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            gains_all_slope_free[station_ids.index(station_id)] = gains_per_run
    
            with open(cal_per_run_no_pulser_removed, "rb") as file:
                cal_per_run = pickle.load(file)

            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            gains_all_no_pulser_removed[station_ids.index(station_id)] = gains_per_run


    #(station, runs)
    dg = [(gains_all_slope_fixed[st_i] - gains_all_slope_free[st_i]) / gains_all_slope_free[st_i] for st_i, _ in enumerate(station_ids)]
    for st_i, dg_st in enumerate(dg):
        for run_i, dg_run in enumerate(dg_st):
            for ch_i, dg_ch in enumerate(dg_run):
                if np.abs(dg_ch) > 0.05:
                    print(f"run {run_nrs[st_i][run_i]}")
                    print(f"station: {station_ids[st_i]}")
                    print(f" channel {channel_ids[ch_i]}")    
    

    # only for use in summary plots THIS IS ALREADY A FLAT LIST
    dg_no_broken_channels = []
    dg_vpols = []
    dg_hpols = []
    dg_LPDA_up = []
    dg_LPDA_down = []
    for st_i, station_id in enumerate(station_ids):
        for channel_id in channel_ids:
            if channel_id in known_broken_channels[str(season)][str(station_id)]:
                continue
            dg_no_broken_channels.extend(dg[station_ids.index(station_id)][:, channel_id])
            if channel_id in channel_types["VPols"]:
                dg_vpols.extend(dg[station_ids.index(station_id)][:, channel_id]) 
            if channel_id in channel_types["HPols"]:
                dg_hpols.extend(dg[station_ids.index(station_id)][:, channel_id]) 
            if channel_id in channel_types["LPDA up"]:
                dg_LPDA_up.extend(dg[station_ids.index(station_id)][:, channel_id]) 
            if channel_id in channel_types["LPDA down"]:
                dg_LPDA_down.extend(dg[station_ids.index(station_id)][:, channel_id]) 

    





    plt.style.use("astroparticle_physics") 
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]




    for station_id in station_ids:
        figname = f"figures/tests/gain_histograms_fitting_slope_fixed_vs_slope_free_st{station_id}" 
        if args.fname_appendix:
            figname += "_" + args.fname_appendix + ".pdf"
        else:
            figname += ".pdf"
        pdf = PdfPages(figname)



        for channel_id in channel_ids:
            fig, ax = plt.subplots()
            ax.hist(
                gains_all_slope_fixed[station_ids.index(station_id)][:, channel_id],
                histtype="stepfilled",
                edgecolor=colors[0],
                facecolor=colors[0] + "88",
                label = "slope fixed"
            )
            ax.hist(
                gains_all_slope_free[station_ids.index(station_id)][:, channel_id],
                histtype="stepfilled",
                edgecolor=colors[1],
                facecolor=colors[1] + "88",
                label = "slope free"
            )

            ax.legend()
            fig.suptitle(f"channel {channel_id}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(
            np.ndarray.flatten(gains_all_slope_fixed[station_ids.index(station_id)]),
            histtype="stepfilled",
            edgecolor=colors[0],
            facecolor=colors[0] + "88",
            label = "slope fixed"
        )
        ax.hist(
            np.ndarray.flatten(gains_all_slope_free[station_ids.index(station_id)]),
            histtype="stepfilled",
            edgecolor=colors[1],
            facecolor=colors[1] + "88",
            label = "slope free"
        )


        ax.legend()
        fig.suptitle(f"all channels")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
        

        pdf.close()


    figname = f"figures/tests/gain_histograms_fitting_slope_fixed_vs_slope_free_difference" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)



    for channel_id in channel_ids:
        dg_channel = [dg[st_i][:, channel_id] for st_i, _ in enumerate(station_ids)]
        fig, ax = plt.subplots()
        ax.hist(
            100 * np.concatenate(dg_channel).ravel(),
            histtype="stepfilled",
            edgecolor=colors[0],
            facecolor=colors[0] + "88"
        )

        ax.set_yscale("log")
        ax.set_xlabel("dGain / %")
        ax.set_ylabel("counts")
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(
        100 * np.array(dg_no_broken_channels),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"all channels")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    

    fig, ax = plt.subplots()
    outlier_edges = [-0.2, 0.2]

    for dg_i, dg_tmp in enumerate(dg_no_broken_channels):
        if dg_tmp < outlier_edges[0]:
            dg_no_broken_channels[dg_i] = outlier_edges[0]
        elif dg_tmp > outlier_edges[-1]:
            dg_no_broken_channels[dg_i] = outlier_edges[-1]
    ax.hist(
        100 * np.array(dg_no_broken_channels),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"all channels with outlier bins")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    
    

    fig, ax = plt.subplots()
    ax.hist(
        100 * np.array(dg_vpols),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"VPols")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)



    fig, ax = plt.subplots()
    ax.hist(
        100 * np.array(dg_hpols),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"HPols")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)



    fig, ax = plt.subplots()
    ax.hist(
        100 * np.array(dg_LPDA_up),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"LPDa up")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)



    fig, ax = plt.subplots()
    ax.hist(
        100 * np.array(dg_LPDA_down),
        histtype="stepfilled",
        edgecolor=colors[0],
        facecolor=colors[0] + "88",
    )

    ax.set_yscale("log")


    ax.set_xlabel("dGain / %")
    ax.set_ylabel("counts")
    fig.suptitle(f"LPDa down")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)

    pdf.close()




    for station_id in station_ids:
        figname = f"figures/tests/gain_histograms_fitting_slope_fixed_vs_slope_free_over_time_st{station_id}" 
        if args.fname_appendix:
            figname += "_" + args.fname_appendix + ".pdf"
        else:
            figname += ".pdf"
        pdf = PdfPages(figname)



        for channel_id in channel_ids:
            fig, ax = plt.subplots()
            ax.scatter(times[station_ids.index(station_id)], gains_all_slope_fixed[station_ids.index(station_id)][:, channel_id],
                       label = "slope fixed")
            ax.scatter(times[station_ids.index(station_id)], gains_all_slope_free[station_ids.index(station_id)][:, channel_id],
                       label = "slope free")
            # ax.scatter(times[station_ids.index(station_id)], gains_all_no_pulser_removed[station_ids.index(station_id)][:, channel_id],
            #            label = "old")
            ax.hlines(default_season_calibration[station_ids.index(station_id)][channel_id],
                      times[station_ids.index(station_id)][0],
                      times[station_ids.index(station_id)][-1],
                      ls="dashed",
                      color=colors[3],
                      lw=2.,
                      label="0 slope"
                      )

            # ax.hlines(slope_fitted_season_calibration[station_ids.index(station_id)][channel_id],
            #           times[station_ids.index(station_id)][0],
            #           times[station_ids.index(station_id)][-1],
            #           ls="dashed",
            #           color=colors[4],
            #           lw=2.,
            #           label="slope fit"
            #           )


            ax.hlines(slope_fitted_weight_normalized_season_calibration[station_ids.index(station_id)][channel_id],
                      times[station_ids.index(station_id)][0],
                      times[station_ids.index(station_id)][-1],
                      ls="dashed",
                      color=colors[5],
                      lw=2.,
                      label="slope fit\nweight normalized"
                      )

            ax.set_xlabel("time")
            ax.set_ylabel("gain / amplitude")
            ax.set_title(f"channel {channel_id}")                       
            ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
            ax.tick_params("x", rotation=-45)

            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)

        for channel_id in channel_ids:
            fig, ax = plt.subplots()
            ax.scatter(times[station_ids.index(station_id)], 100 * dg[station_ids.index(station_id)][:, channel_id],
                       label = "slope fixed")
            ax.tick_params("x", rotation=-45)
            ax.set_xlabel("time")
            ax.set_ylabel("dg / %")
            ax.set_title(f"channel {channel_id}")                       
            ax.legend(loc="best")

            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)
        


        pdf.close()