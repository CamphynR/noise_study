import argparse
import datetime
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pandas as pd
import pickle
from scipy.stats import gaussian_kde

from rnog_data.runtable import RunTable
from utilities.utility_functions import convert_to_db, convert_error_to_db



def plot_dG_histogram(run_times_all_flattened, dG_all_flattened,
                      cmap="YlOrRd",
                      gain_lim=[-30, 30],
                      gain_bin_width=1,
                      density=False,
                      baseline_label="yearly",
                      fixed_baseline=None,
                      fname_appendix=None
                      ):

    fig, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, sharey=True, facecolor="white")
    plt.subplots_adjust(wspace=0.1, bottom=0.25, top=0.95, right=0.83)

    ax_left_xlim = [np.datetime64("2022-07-01T00:00:00").astype("datetime64[ns]").astype("float64"),
                    np.datetime64("2022-09-01T00:00:00").astype("datetime64[ns]").astype("float64")]
    ax_middle_xlim = [np.datetime64("2023-02-20T00:00:00").astype("datetime64[ns]").astype("float64"),
                      np.datetime64("2023-12-31T00:00:00").astype("datetime64[ns]").astype("float64")]
    ax_right_xlim = [np.datetime64("2024-04-01T00:00:00").astype("datetime64[ns]").astype("float64"),
                     np.datetime64("2024-08-01T00:00:00").astype("datetime64[ns]").astype("float64"),]


    gain_bins = np.arange(gain_lim[0], gain_lim[1] + gain_bin_width, gain_bin_width)
#    gain_bins = 50

#    ax_left_bins = np.linspace(*ax_left_xlim, 50)
#    ax_middle_bins = np.linspace(*ax_middle_xlim, 50)
#    ax_right_bins = np.linspace(*ax_right_xlim, 50)

    ax_left_bins = 200 
    ax_middle_bins = ax_left_bins
    ax_right_bins = ax_left_bins

    # this sets empty bins to be white
    cmin=1e-30
    vmin=1e-30


    # define overflow bins
    for dG_idx, dG_tmp in enumerate(dG_all_flattened):
        if dG_tmp * 100 > gain_lim[-1]:
            dG_all_flattened[dG_idx] = gain_bins[-1] / 100
        elif dG_tmp * 100 < gain_lim[0]:
            dG_all_flattened[dG_idx] = gain_bins[0] / 100


    ax_left.hist2d(np.array(run_times_all_flattened).astype("float64"),
                   np.array(dG_all_flattened) * 100,
                   bins=(ax_left_bins, gain_bins),
                   cmap=cmap,
                   cmin=cmin,
                   vmin=vmin,
                   density=density)
    ax_middle.hist2d(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100,
                     bins=(ax_middle_bins, gain_bins),
                   cmap=cmap,
                   cmin=cmin,
                     vmin=vmin,
                     density=density)
    hist, x_bin, y_bin, img= ax_right.hist2d(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100,
                    bins=(ax_right_bins, gain_bins),
                   cmap=cmap,
                   cmin=cmin,
                                             vmin=vmin,
                    density=density)

    min_run_time = min([min(t.astype("float64")) for r in run_times_all.values() for t in r.values()])
    max_run_time = max([max(t.astype("float64")) for r in run_times_all.values() for t in r.values()])


    baseline_label = f"Baseline gain\nseason {baseline_label}"
    if args.fixed_baseline:
        baseline_label = f"Baseline {args.fixed_baseline} dB"
    
    for ax in [ax_left, ax_middle, ax_right]:
        ax.hlines(0,
                  min_run_time,
                  max_run_time,
                  ls="dashed",
                  lw=1.5,
                  color="black",
                  label=baseline_label)


    # size of diagonal marks
    d = .015

    kwargs = dict(transform=ax_left.transAxes,
                  color='k',
                  clip_on=False)

    # right side of left axis
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax_middle.transAxes)

    # left side of middle axis
    ax_middle.plot((-d, +d), (-d, +d), **kwargs)
    ax_middle.plot((-d, +d), (1-d, 1+d), **kwargs)

    # right side of middle axis
    ax_middle.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_middle.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax_right.transAxes)

    # left side of right axis
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)


    ax_left.set_xlim(*ax_left_xlim)
    ax_middle.set_xlim(*ax_middle_xlim)
    ax_right.set_xlim(*ax_right_xlim)
    ax_left.spines["right"].set_visible(False)
    ax_middle.spines["left"].set_visible(False)
    ax_middle.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_left.yaxis.tick_left()


    formatter = FuncFormatter(lambda x, tick_pos: x.astype("datetime64[ns]").astype("datetime64[D]"))
    for ax in [ax_left, ax_middle, ax_right]:
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-35, ha="left")
    

    cbar = plt.colorbar(img, ax=ax_right)
#    cbar_ticks = cbar.get_ticks()
    # renormalize the pdf to use times starting from first run in seconds
#    cbar.set_ticks(cbar_ticks[:-1], labels = [f"{lab:.2f} %" for lab in 100 * np.array(cbar_ticks[:-1])*np.diff(x_bin)[0]*np.diff(y_bin)[0]])
    cbar.set_label("# runs", rotation=-90, labelpad=15)
    

    ax_right.legend(loc="lower right")
    ax_left.set_ylabel("dG / %")


    figname = "figures/calibration_summaries/spread_dG_all"
    if fname_appendix:
        figname += "_" + f"{fname_appendix}"
    figname += ".png"
    print(figname)
    try:
        fig.savefig(figname, dpi=150)
    except ValueError:
        print(dG_all_flattened)
        print(hist)
        exit()
    plt.close(fig)




def construct_flattened_hist_arrays(run_times_all, dG_all, station_ids, channel_ids):
    run_times_all_flattened_tmp = []
    for season, run_times_season in run_times_all.items():
        for station_id, run_times_station in run_times_season.items():
            if int(station_id) not in station_ids:
                continue
            run_times_all_flattened_tmp.extend(np.tile(run_times_station, len(channel_ids)))


    dG_all_flattened_tmp = []
    for season, dG_season in dG_all.items():
        for station_id, dG_station in dG_season.items():
            if int(station_id) not in station_ids:
                continue
            for channel_id, dG_channel in dG_station.items():
                if channel_id not in channel_ids:
                    continue
                dG_all_flattened_tmp.extend(dG_channel)
    return run_times_all_flattened_tmp, dG_all_flattened_tmp

def construct_baseline(baseline_type,
                       seasons, station_ids
                       ):
    
    baseline = {}
    if baseline_type == "yearly":
        for season in seasons:
            if season == "2024":
                season = "2024_radiant_v2"
            baseline[season] = {}
            for station_id in station_ids:
                
                # no data for these stations in first part of 2024
                if station_id in [21, 22] and season=="2024_radiant_v2":
                    continue

                print("YOU ARE USING DEFAULT NO PULSER DO NOT FORGET")
                baseline_calibration_path=f"absolute_amplitude_results/season{season}/station{station_id}/default_no_pulser_removed/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
                baseline_calibration = pd.read_csv(baseline_calibration_path,
                                                  index_col=0)
                baseline[season][station_id] = baseline_calibration["gain"]

    elif baseline_type.isdigit():
        season = baseline_type
        baseline[season] = {}
        for station_id in station_ids:
            print("YOU ARE USING DEFAULT NO PULSER DO NOT FORGET")
            baseline_calibration_path=f"absolute_amplitude_results/season{season}/station{station_id}/default_no_pulser_removed/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
            baseline_calibration = pd.read_csv(baseline_calibration_path,
                                              index_col=0)
            baseline[season][station_id] = baseline_calibration["gain"]

    return baseline
    


def select_baseline_gain(run_time, baseline, baseline_type, station_id, channel_id):
    if type(station_id) is str:
        station_id = int(station_id)
    if baseline_type == "yearly":
        season = run_time.astype("datetime64[Y]").astype(str)
        if season == "2024":
            season = "2024_radiant_v2"
        return baseline[season][station_id][channel_id]
    elif baseline_type.isdigit():
        return baseline[baseline_type][station_id][channel_id]





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_files",
                        help="path to pickle file containing gain per run",
                        nargs="+")
#    parser.add_argument("--default_calibration", default=None,
#                        help="path to default calibration (over full season)\
#                                if None the code tries to infer the path from args.summary_file")
    parser.add_argument("--baseline", default=None,
                        help="season with respect to which calculate difference in calibration,\
                                if None, 2023 will be used\
                                can also choose different baseline time bins, options are\
                                yearly, (to be added)")
    parser.add_argument("--fixed_baseline", type=float, default=None,
                        help="fixed baseline for all channels, give in dB")
    parser.add_argument("--fname_appendix")
    args = parser.parse_args()



    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)



    seasons = []
    station_ids = []

    for args.summary_file in args.summary_files:
        if "season" in args.summary_file:
            season = args.summary_file.split("season")[1][:4]
            if season == "2024":
                season = "2024_radiant_v2"
            seasons.append(season)

        if "station" in args.summary_file:
            station_id = args.summary_file.split("station")[1][:2]
            station_ids.append(station_id)

    # manualy selecting to avoid station 22, temp
    # station_ids = [11, 12, 13, 21, 22, 23, 24]
#    station_ids = [13]


    run_times_all = {season_tmp : {} for season_tmp in seasons}
    dG_all = {season_tmp : {} for season_tmp in seasons}

    

    if args.fixed_baseline:
        fixed_baseline = 10**(args.fixed_baseline / 20.)


    baseline = construct_baseline(args.baseline, seasons, station_ids)
    

    
    plt.style.use("astroparticle_physics")
    colors = ["red", "blue", "black", "orange", "yellow", "purple", "green",  "pink"]
    pdf_path = "figures/calibration_summaries/calibration_per_run_all.pdf"
    pdf = PdfPages(pdf_path)
    fig, ax = plt.subplots()

    for args.summary_file in args.summary_files:
        if "season" in args.summary_file:
            season = args.summary_file.split("season")[1][:4]
            if season == "2024":
                season = "2024_radiant_v2"

        if "station" in args.summary_file:
            station_id = args.summary_file.split("station")[1][:2]

        if int(station_id) not in station_ids:
            continue


        channel_ids = np.arange(24)
#        channel_ids = [4]



        



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
        dG_dict = {}
        for channel_id in channel_ids:
            dG_run = []
            for run_i, dG_tmp in enumerate(gain_per_run[:, channel_id]):

                run_time = run_times[run_i]
                baseline_gain = select_baseline_gain(run_time, baseline, args.baseline, station_id, channel_id)
                dG_run.append((dG_tmp - baseline_gain) / baseline_gain)
                if args.fixed_baseline:
                    dG_run.append((gain_per_run[:, channel_id] - fixed_baseline) / fixed_baseline)
            dG_run = np.array(dG_run)

            dG.append(dG_run)
            dG_dict[channel_id] = dG_run
        dG = np.array(dG)
        
        dG_trimmed = []
        for dG_tmp in dG:
            if channel_id in known_broken_channels[season][station_id]:
                continue
            dG_trimmed.extend(dG)
#        dG_all[season][station_id] = dG
        dG_all[season][station_id] = dG_dict

        for channel_id in channel_ids:
            if channel_id in known_broken_channels[season][station_id]:
                continue
            
            if season == "2022" and channel_id == 1:
                label = f"station {station_id}"
            else:
                label = None

            ax.scatter(run_times, dG_dict[channel_id] * 100,
                       rasterized=True,
                       color=colors[station_ids.index(int(station_id))],
                       label=label)

        
    baseline_label = f"Baseline gain\n {args.baseline}"
    if args.fixed_baseline:
        baseline_label = f"Baseline {args.fixed_baseline} dB"
    ax.hlines(0,
              min([min(t) for r in run_times_all.values() for t in r.values()]),
              max([max(t) for r in run_times_all.values() for t in r.values()]),
              ls="dashed",
              lw=1.5,
              color="black",
              label=baseline_label)

    ax.tick_params("x", rotation=30)
#    ax.set_ylim(-20, 40)
    ax.set_xlabel("date / local time")
    ax.set_ylabel("dG / %")
    ax.legend(loc="upper left", bbox_to_anchor=(1.,1.))



    fig.tight_layout()
    fig.savefig(pdf_path.split(".pdf")[0], dpi=300)
    fig.savefig(pdf, format="pdf")
    pdf.close()


    # plot everything
#    station_ids = [11, 12, 13, 21, 22, 23, 24]
    channel_ids = np.arange(24)

    run_times_all_flattened, dG_all_flattened = construct_flattened_hist_arrays(run_times_all, dG_all,
                                                                                station_ids, channel_ids)
    plot_dG_histogram(run_times_all_flattened,
                      dG_all_flattened,
                      baseline_label=args.baseline,
                      )



    channel_groups = {
            "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA up" :[13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20]
            }

    for channel_group_name, channel_group in channel_groups.items():
        run_times_all_flattened, dG_all_flattened = construct_flattened_hist_arrays(run_times_all, dG_all,
                                                                                    station_ids, channel_ids=channel_group)
        plot_dG_histogram(run_times_all_flattened,
                          dG_all_flattened,
                          baseline_label=args.baseline,
                          fname_appendix=f"{channel_group_name}"
                          )
        for station_id in station_ids:
            run_times_all_flattened, dG_all_flattened = construct_flattened_hist_arrays(run_times_all, dG_all,
                                                                                        station_ids=[station_id], channel_ids=channel_group)
            plot_dG_histogram(run_times_all_flattened,
                              dG_all_flattened,
                                baseline_label=args.baseline,
                              fname_appendix=f"st{station_id}_{channel_group_name}"
                              )

    
    for station_id in station_ids:
        run_times_all_flattened, dG_all_flattened = construct_flattened_hist_arrays(run_times_all, dG_all,
                                                                                    station_ids=[station_id], channel_ids=channel_ids)
        plot_dG_histogram(run_times_all_flattened,
                          dG_all_flattened,
                          baseline_label=args.baseline,
                          fname_appendix=f"st{station_id}"
                          )



    channel_ids = np.arange(24)
    for station_id in station_ids:
        for channel_id in channel_ids:
            run_times_all_flattened, dG_all_flattened = construct_flattened_hist_arrays(run_times_all, dG_all,
                                                                                        station_ids=[station_id], channel_ids=[channel_id])
            plot_dG_histogram(run_times_all_flattened,
                              dG_all_flattened,
                          baseline_label=args.baseline,
                              fname_appendix=f"st{station_id}_ch_{channel_id}"
                              )
    
#    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, facecolor="white")
#
##    xy = np.vstack([np.array(run_times_all_flattened).astype("float64"),
##                    np.array(dG_all_flattened) * 100])
##    z = gaussian_kde(xy)(xy)
#
#    ax_left.scatter(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, c="red", s=100)
#    ax_right.scatter(np.array(run_times_all_flattened).astype("float64"), np.array(dG_all_flattened) * 100, c="red", s=100)
#
#    min_run_time = min([min(t.astype("float64")) for r in run_times_all.values() for t in r.values()])
#    max_run_time = max([max(t.astype("float64")) for r in run_times_all.values() for t in r.values()])
#
#
#    ax_left.hlines(0,
#                   min_run_time,
#                   max_run_time,
#                  ls="dashed",
#                  lw=1.5,
#                  color="black",
#                  label=f"Baseline gain\nseason {baseline_season}")
#    ax_right.hlines(0,
#                   min_run_time,
#                   max_run_time,
#                  ls="dashed",
#                  lw=1.5,
#                  color="black",
#                  label=f"Baseline gain\nseason {baseline_season}")
#
#    ax_left.set_xlim(None, np.datetime64("2022-09-01T00:00:00").astype("datetime64[ns]").astype("float64"))
#    ax_right.set_xlim(np.datetime64("2023-02-20T00:00:00").astype("datetime64[ns]").astype("float64"), None)
#    ax_left.spines["right"].set_visible(False)
#    ax_right.spines["left"].set_visible(False)
##    ax_right.yaxis.set_ticks_position("right")
#    ax_left.yaxis.tick_left()
##    ax_left.tick_params(labelright="off")
##    ax_right.yaxis.tick_right()
#
##    ax_left.set_ylim(-20, 20)
#
#    formatter = FuncFormatter(lambda x, tick_pos: x.astype("datetime64[ns]").astype("datetime64[D]"))
#    for ax in (ax_left, ax_right):
#        ax.xaxis.set_major_formatter(formatter)
#        ax.tick_params("x", rotation=30)
#
#    ax_right.legend(loc="lower right")
#    ax_left.set_ylabel("dG / %")
#    fig.tight_layout()
#    fig.savefig("figures/calibration_summaries/spread_dG_all_scatter.png", dpi=150)
