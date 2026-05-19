import argparse
import datetime
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pickle
from scipy.stats import gaussian_kde

from rnog_data.runtable import RunTable
from utilities.utility_functions import convert_to_db, convert_error_to_db






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_files",
                        help="path to pickle file containing gain per run",
                        nargs="+")
    parser.add_argument("--default_calibration", default=None,
                        help="path to default calibration (over full season)\
                                if None the code tries to infer the path from args.summary_file")
    parser.add_argument("--baseline_season", default=None,
                        help="season with respect to which calculate difference in calibration,\
                                if None, 2023 will be used")
    args = parser.parse_args()



    facealpha = 0.2
    colors_face = [(205./255, 146./255, 218./255, facealpha), # pink
                   (82./255, 27./255, 241./255, facealpha),   # blue
                   (190./255, 0., 0., facealpha),             # red
                   (238./255, 206./255, 0., facealpha),       # yellow
                   (33./255, 175./255, 0., facealpha),        # green
                   (110./255, 18./255, 177./255, facealpha),  # purple
                   (110./255, 183./255, 203./255, facealpha), # turquose
                   (237./255, 103./255, 40./255, facealpha),   # orange
                   (10./255, 10./255, 10./255, facealpha)   # black
                   ]
                           

    edgealpha = 1.
    colors_edge = [(205./255, 146./255, 218./255, edgealpha), # pink
                   (82./255, 27./255, 241./255, edgealpha),   # blue
                   (190./255, 0., 0., edgealpha),             # red
                   (238./255, 206./255, 0., edgealpha),       # yellow
                   (33./255, 175./255, 0., edgealpha),        # green
                   (110./255, 18./255, 177./255, edgealpha),  # purple
                   (110./255, 183./255, 203./255, edgealpha), # turquose
                   (237./255, 103./255, 40./255, edgealpha),   # orange
                   (10./255, 10./255, 10./255, edgealpha)   # black
                   ]

    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pdf_path = "figures/calibration_summaries/calibration_per_run_all.pdf"
    pdf = PdfPages(pdf_path)
    fig, ax = plt.subplots()

    run_times_all = {"2022" : {}, "2023" : {}}
    dG_all = {"2022": {}, "2023" : {}}
    stations = [11, 12, 13, 21, 22, 23, 24]

    if args.baseline_season:
        baseline_season = args.baseline_season
    else:
        baseline_season = 2023
       
    
    for args.summary_file in args.summary_files:
        if "season" in args.summary_file:
            season = args.summary_file.split("season")[1][:4]

        if "station" in args.summary_file:
            station_id = args.summary_file.split("station")[1][:2]


        channel_ids = np.arange(24)


        if args.default_calibration is None:
            default_calibration_path = f"absolute_amplitude_results/season{baseline_season}/station{station_id}/default/absolute_amplitude_calibration_season{baseline_season}_st{station_id}_best_fit.csv"
            default_calibration_error_path = f"absolute_amplitude_results/season{baseline_season}/station{station_id}/default/absolute_amplitude_calibration_season{baseline_season}_st{station_id}_best_fiterror.csv"
        else:
            default_calibration_path = args.default_calibration


        default_calibration = pd.read_csv(default_calibration_path,
                                          index_col=0)
        default_calibration_error = pd.read_csv(default_calibration_error_path,
                                                index_col=0)
        



        with open(args.summary_file, "rb") as file:
            calibration_per_run = pickle.load(file)

        run_nrs = calibration_per_run["run_nr"]
        gain_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in calibration_per_run["calibration"]])



        rt = RunTable()
        runtable_kwargs = dict(
                stations=[int(station_id)],
                runs=run_nrs,
                run_types=["physics"]
                )
        rnog_table = rt.get_table(**runtable_kwargs)
        run_times = rnog_table["time_start"].to_numpy()
        run_times_all[season][station_id] = run_times







        dG = []
        for channel_id in channel_ids:
            default_gain = default_calibration["gain"][channel_id]
            dG_run = (gain_per_run[:, channel_id] - default_gain) / default_gain
            dG.append(dG_run)
        dG = np.array(dG)
        
        dG_trimmed = []
        for dG_tmp in dG:
            if channel_id in known_broken_channels[season][station_id]:
                continue
            dG_trimmed.extend(dG)
        dG_all[season][station_id] = dG

        for channel_id in channel_ids:
            if channel_id in known_broken_channels[season][station_id]:
                continue
            
            if season == "2022" and channel_id == 1:
                label = f"station {station_id}"
            else:
                label = None

            ax.scatter(run_times, dG[channel_id] * 100,
                       color=colors_edge[stations.index(int(station_id))],
                       label=label)

        
    ax.hlines(0,
              min([min(t) for r in run_times_all.values() for t in r.values()]),
              max([max(t) for r in run_times_all.values() for t in r.values()]),
              ls="dashed",
              lw=1.5,
              color="black",
              label=f"Baseline gain\nseason {baseline_season}")

    ax.tick_params("x", rotation=30)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("date / local time")
    ax.set_ylabel("dG / %")
    ax.legend(loc="upper left", bbox_to_anchor=(1.,1.))



    fig.tight_layout()
    fig.savefig(pdf_path.split(".pdf")[0], dpi=300)
    fig.savefig(pdf, format="pdf")
    pdf.close()

    run_times_all_flattened = []
    for run_times_season in run_times_all.values():
        for run_times_station in run_times_season.values():
            run_times_all_flattened.extend(np.tile(run_times_station, len(channel_ids)))


    dG_all_flattened = []
    for dG_season in dG_all.values():
        for dG_station in dG_season.values():
            for dG_channel in dG_station:
                dG_all_flattened.extend(dG_channel)


    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, facecolor="white")
    ax_left.hist2d(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, bins=(50, 1000))
    ax_right.hist2d(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, bins=(50, 1000))

    min_run_time = min([min(t.astype("float64")) for r in run_times_all.values() for t in r.values()])
    max_run_time = max([max(t.astype("float64")) for r in run_times_all.values() for t in r.values()])


    ax_left.hlines(0,
                   min_run_time,
                   max_run_time,
                  ls="dashed",
                  lw=1.5,
                  color="black",
                  label=f"Baseline gain\nseason {baseline_season}")
    ax_right.hlines(0,
                   min_run_time,
                   max_run_time,
                  ls="dashed",
                  lw=1.5,
                  color="black",
                  label=f"Baseline gain\nseason {baseline_season}")

    ax_left.set_xlim(None, np.datetime64("2022-09-01T00:00:00").astype("datetime64[ns]").astype("float64"))
    ax_right.set_xlim(np.datetime64("2023-02-20T00:00:00").astype("datetime64[ns]").astype("float64"), None)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.set_ticks_position("right")
    ax_left.yaxis.tick_left()
#    ax_left.tick_params(labelright="off")
    ax_right.yaxis.tick_right()

    ax_left.set_ylim(-20, 20)

    formatter = FuncFormatter(lambda x, tick_pos: x.astype("datetime64[ns]").astype("datetime64[D]"))
    for ax in (ax_left, ax_right):
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params("x", rotation=30)

    ax_right.legend(loc="lower right")
    ax_left.set_ylabel("dG / %")
    fig.tight_layout()
    fig.savefig("figures/calibration_summaries/spread_dG_all.png", dpi=150)



    
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, facecolor="white")

    xy = np.vstack([np.array(run_times_all_flattened).astype("float64"),
                    np.array(dG_all_flattened) * 100])
    z = gaussian_kde(xy)(xy)

    ax_right.scatter(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, c=z, s=100)
    ax_right.scatter(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, c=z, s=100)

    min_run_time = min([min(t.astype("float64")) for r in run_times_all.values() for t in r.values()])
    max_run_time = max([max(t.astype("float64")) for r in run_times_all.values() for t in r.values()])


    ax_left.hlines(0,
                   min_run_time,
                   max_run_time,
                  ls="dashed",
                  lw=1.5,
                  color="black",
                  label=f"Baseline gain\nseason {baseline_season}")
    ax_right.hlines(0,
                   min_run_time,
                   max_run_time,
                  ls="dashed",
                  lw=1.5,
                  color="black",
                  label=f"Baseline gain\nseason {baseline_season}")

    ax_left.set_xlim(None, np.datetime64("2022-09-01T00:00:00").astype("datetime64[ns]").astype("float64"))
    ax_right.set_xlim(np.datetime64("2023-02-20T00:00:00").astype("datetime64[ns]").astype("float64"), None)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.set_ticks_position("right")
    ax_left.yaxis.tick_left()
#    ax_left.tick_params(labelright="off")
    ax_right.yaxis.tick_right()

    ax_left.set_ylim(-20, 20)

    formatter = FuncFormatter(lambda x, tick_pos: x.astype("datetime64[ns]").astype("datetime64[D]"))
    for ax in (ax_left, ax_right):
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params("x", rotation=30)

    ax_right.legend(loc="lower right")
    ax_left.set_ylabel("dG / %")
    fig.tight_layout()
    fig.savefig("figures/calibration_summaries/spread_dG_all_scatter.png", dpi=150)
