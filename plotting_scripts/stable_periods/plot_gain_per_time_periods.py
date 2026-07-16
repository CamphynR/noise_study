import argparse
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



# def compare_var(gains, nr_stable_gains, gain_idx, channel_id):
#     mean = np.mean(gains[gain_idx - nr_stable_gains : gain_idx - 1, channel_id])
#     var = np.var(gains[gain_idx - nr_stable_gains : gain_idx-1, channel_id])
#     metric = np.abs(gains[gain_idx, channel_id] - mean)/var
#     return metric



def define_stable_periods_based_on_windows(gains,
                          nr_stable_gains=48,
                          threshold=0.05,
                          channel_ids=np.arange(24),
                          metric_function=None):
    
    split_idxs = [[] for _ in channel_ids]
    metric_all = [[] for _ in channel_ids]
    for channel_id in channel_ids:
        prev_sliding_mean_gain = sliding_mean(gains, nr_stable_gains, 0, channel_id)
        for gain_idx, gain in enumerate(gains):
            if gain_idx == 0:
                continue
            sliding_mean_gain = sliding_mean(gains, nr_stable_gains, gain_idx, channel_id)

            metric = np.abs(sliding_mean_gain - prev_sliding_mean_gain) / prev_sliding_mean_gain
            if metric > threshold:
                split_idxs[channel_id].append(gain_idx)
        

            metric_all[channel_id].append(metric)
    

    metric_all = np.array(metric_all)

    return split_idxs, metric_all



def define_stable_periods(gains,
                          nr_stable_gains=48,
                          threshold=0.05,
                          channel_ids=np.arange(24),
                          metric_function=None):
    "returns indices in list of gains where a stable period ends"
    
    split_idxs = [[] for _ in channel_ids]
    metric_all = [[] for _ in channel_ids]
    outlier_indices = [[] for _ in channel_ids]
    for channel_id in channel_ids:
        # nr_stable_gains = 3
        for gain_idx, gain in enumerate(gains):
            if gain_idx == 0:
                continue

            if gain_idx < nr_stable_gains or gain_idx > len(gains) - nr_stable_gains:
                metric_all[channel_id].append(0)
                continue
                # if np.abs(gain[channel_id] - gains[gain_idx - 1, channel_id])/gains[gain_idx - 1, channel_id] > threshold:
                #     split_idxs[channel_id].append(gain_idx)

            if metric_function is None:
                gains_mean = np.mean(gains[gain_idx - nr_stable_gains : gain_idx + nr_stable_gains, channel_id])
                metric = np.abs(gain[channel_id] - gains_mean)/gains_mean
            else:
                metric = metric_function(gains, nr_stable_gains, gain_idx, channel_id)

            metric_all[channel_id].append(metric)

#            if channel_id == 0:
#                if gain_idx < 500 and gain_idx > 200: 
#                    print(gain_mean)
#                    print(gain[channel_id])
#                    print(np.abs(gain[channel_id] - gain_mean)/gain_mean)


            if  metric > threshold:
                # test whether you are an outlier
                # if np.abs(gains[gain_idx + 1, channel_id] - gain[channel_id]) / gain[channel_id] > threshold and np.abs(gains[gain_idx - 1, channel_id] - gain[channel_id]) / gain[channel_id] > threshold:
                #     outlier_indices[channel_id].append(gain_idx)
                #     continue
                # else:
                # nr_stable_gains = 3
                split_idxs[channel_id].append(gain_idx)

            # nr_stable_gains += 1

    metric_all = np.array(metric_all)

    return split_idxs, metric_all

        




    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_appendix", default=None)
    args = parser.parse_args()
    seasons = [2023]
    station_id = 11
    channel_ids = np.arange(24)
    plot_relative = False
    

    rt = RunTable()
    runtable_kwargs = dict(
            stations=[station_id],
            start_time=f"{seasons[0]}-01-01",
            stop_time=f"{seasons[-1]}-12-31",
            run_types=["physics"]
            )
    table = rt.get_table(**runtable_kwargs)

    forced_trigger_idx = table["trigger_soft_enabled"] == 1
    table = table[forced_trigger_idx]



    # seasonal calibration used for these runs per season
    calibration_season_path = f"absolute_amplitude_results/season{seasons[0]}/station{station_id}/default/absolute_amplitude_calibration_season{seasons[0]}_st{station_id}_best_fit.csv"
    calibration_season = pd.read_csv(calibration_season_path, index_col=0)
    gain_season = calibration_season["gain"]



    vrms_all = []
    times = []
    gains_all = []
    period_indices = [[] for _ in channel_ids]
    for season in seasons:
        data_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.2/season{season}/station{station_id}/clean/"
        
        data_paths = natsorted([f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.2/season{season}/station{station_id}/clean/{filename}" for filename in os.listdir(data_dir)])

        vrms = []
        var_vrms = []
        for pickle_file in data_paths:
            rms_dict = read_pickle(pickle_file)
            vrms.append(rms_dict["vrms"])
            var_vrms.append(rms_dict["var_vrms"])
        vrms = np.array(vrms).T
        var_vrms = np.array(var_vrms).T
        vrms_all.extend(vrms)


        cal_per_run_path = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season2023/station11/slope_fixed_to_2023/season2023_st11_all_runs_compiled_slope_fixed_to_2023.pickle"

        with open(cal_per_run_path, "rb") as file:
            cal_per_run = pickle.load(file)

        table_season = table[table["run"].isin(cal_per_run["run_nr"])]
        times.extend(table_season["time_start"])


        gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
        gains_all.extend(gains_per_run)
    


    gains_all = np.array(gains_all)
    # metric_function = per_run_metric
    # metric_function = compare_var
    # metric_function = compare_means
    # metric_function = compare_to_mean
    metric_function = None
    period_indices_tmp, metric_all = define_stable_periods_based_on_windows(gains_all, metric_function=metric_function)
    for channel_id in channel_ids:
        period_indices[channel_id].extend(period_indices_tmp[channel_id])
    times = np.array(times)
    vrms_all = np.array(vrms_all)



    plt.style.use("astroparticle_physics") 
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figname = f"figures/stable_gain_periods_st{station_id}" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)



    for channel_id in channel_ids:
        fig, axs = plt.subplots(2, 1, sharex=True)
        if plot_relative:
            dG = 100 * np.diff(gains_all[:, channel_id]) / gains_all[:-1, channel_id]
             
            axs[0].scatter(times[1:], dG,
                           s=1.,
                           )
            axs[0].scatter(times[period_indices[channel_id][1:]], dG[period_indices[channel_id][1:]],
                           s=3.,
                           color=colors[3])
            axs[0].vlines(times[period_indices[channel_id][1:]],
                          -np.max(dG),
                          np.max(dG),
                          ls="dashed",
                          color=colors[1]
                          )
            axs[0].set_ylim(-10,
                            10)
            axs[0].set_ylabel("dGain / %")
        else:
            axs[0].scatter(times, gains_all[:, channel_id],
                           s=1.,
                           )
            axs[0].hlines(gain_season[channel_id], times[0], times[-1],
                          ls="dashed", color=colors[0])

            axs[0].set_ylim(0.95 * np.min(gains_per_run[:, channel_id]),
                            1.05 * np.max(gains_per_run[:, channel_id]))
                
            axs[0].vlines(times[period_indices[channel_id]],
                          0,
                          2*np.max(gains_per_run[channel_id]),
                          ls="dashed",
                          color=colors[1]
                          )
            axs[0].set_ylabel("Gain / amplitude")
#        axs[0].set_xlim(
#            times[420],
#            times[480],
#            )


        axs[1].scatter(times, vrms_all[channel_id]/units.mV,
                       s=1.)
        axs[1].vlines(times[period_indices[channel_id]],
                      0,
                      2*np.max(vrms_all[channel_id]) / units.mV,
                      ls="dashed",
                      color=colors[1],
                      label=f"nr of periods: {len(period_indices[channel_id])}"
                      )
        axs[1].set_ylim(0.95 * np.min(vrms_all[channel_id]/units.mV),
                        1.05 * np.max(vrms_all[channel_id]/units.mV))
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("Vrms / mV")
        axs[1].tick_params(axis="x", rotation = -45)
        axs[1].legend()


        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()



    figname = f"figures/stable_gain_periods_st{station_id}_metric_hist" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.hist(metric_all[channel_id] * 100,
                histtype="stepfilled",
                facecolor=colors[0] + "88",
                edgecolor=colors[0],
                lw=3.
                )


        ax.set_yscale("log")
        ax.set_xlabel("metric / %")
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.ndarray.flatten(metric_all) * 100,
            histtype="stepfilled",
            facecolor=colors[0] + "88",
            edgecolor=colors[0],
            lw=3.
            )
    ax.set_xlabel("metric / %")
    ax.set_yscale("log")


    fig.suptitle("all channels")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()


    figname = f"figures/stable_gain_periods_st{station_id}_metric_over_time" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        print(len(times))
        print(len(metric_all[channel_id]))
        ax.scatter(
            times[1:],
            metric_all[channel_id],
            s=1.
            )


        ax.set_yscale("log")
        ax.set_xlabel("metric / %")
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()



    figname = f"figures/stable_gain_periods_st{station_id}_metric_hist" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.hist(metric_all[channel_id] * 100,
                histtype="stepfilled",
                facecolor=colors[0] + "88",
                edgecolor=colors[0],
                lw=3.
                )


        ax.set_yscale("log")
        ax.set_xlabel("metric / %")
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.ndarray.flatten(metric_all) * 100,
            histtype="stepfilled",
            facecolor=colors[0] + "88",
            edgecolor=colors[0],
            lw=3.
            )
    ax.set_xlabel("metric / %")
    ax.set_yscale("log")


    fig.suptitle("all channels")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()



    figname = f"figures/gains_per_run_histogram_st{station_id}" 
    if args.fname_appendix:
        figname += "_" + args.fname_appendix + ".pdf"
    else:
        figname += ".pdf"
    pdf = PdfPages(figname)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.hist(gains_all[:, channel_id],
                histtype="stepfilled",
                facecolor=colors[0] + "88",
                edgecolor=colors[0],
                lw=3.
                )


        ax.set_yscale("log")
        ax.set_xlabel("gains / amp")
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.ndarray.flatten(gains_all),
            histtype="stepfilled",
            facecolor=colors[0] + "88",
            edgecolor=colors[0],
            lw=3.
            )
    ax.set_xlabel("gains / amp")
    ax.set_yscale("log")


    fig.suptitle("all channels")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()
