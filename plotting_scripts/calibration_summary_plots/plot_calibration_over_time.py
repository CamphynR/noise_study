import argparse
import datetime
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


from rnog_data.runtable import RunTable
from utilities.utility_functions import convert_to_db, convert_error_to_db






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_file",
                        help="path to pickle file containing gain per run")
    parser.add_argument("--default_calibration", default=None,
                        help="path to default calibration (over full season)\
                                if None the code tries to infer the path from args.summary_file")
    args = parser.parse_args()


    
    if "season" in args.summary_file:
        season = args.summary_file.split("season")[1][:4]

    if "station" in args.summary_file:
        station_id = args.summary_file.split("station")[1][:2]


    channel_ids = np.arange(24)


    if args.default_calibration is None:
        default_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
        default_calibration_error_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fiterror.csv"
    else:
        default_calibration_path = args.default_calibration


    default_calibration = pd.read_csv(default_calibration_path,
                                      index_col=0)
    default_calibration_error = pd.read_csv(default_calibration_error_path,
                                            index_col=0)
    

    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


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


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pdf_path = f"figures/calibration_summaries/gain_per_run_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_path)


    for channel_id in channel_ids:
        if channel_id in known_broken_channels[season][station_id]:
            continue
        fig, ax = plt.subplots()
        ax.scatter(run_times,
                   convert_to_db(gain_per_run[:, channel_id]),
                   label="calibration per run")
        ax.hlines(convert_to_db(default_calibration["gain"][channel_id]),
                  run_times[0], run_times[-1],
                  lw=2.,
                  color=colors[1],
                  label="default calibration")
        ax.fill_between(
                run_times,
                convert_to_db(default_calibration["gain"][channel_id] - default_calibration_error["gain"][channel_id]),
                convert_to_db(default_calibration["gain"][channel_id] + default_calibration_error["gain"][channel_id]),
                color=colors[1],
                alpha=0.3
                )
        ax.legend()
        ax.set_xlabel("run time")
        ax.set_ylabel("Gain / dB")
        ax.tick_params("x", rotation=30)
        ax.set_title(f"Channel {channel_id}")
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)




    pdf.close()

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





    dG = []
    for channel_id in channel_ids:
        default_gain = default_calibration["gain"][channel_id]
        dG_run = (gain_per_run[:, channel_id] - default_gain) / default_gain
        dG.append(dG_run)
    dG = np.array(dG)

    pdf_path = f"figures/calibration_summaries/dG_per_run_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_path)
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        if channel_id in known_broken_channels[season][station_id]:
            continue
        ax.scatter(run_times, dG[channel_id] * 100)
        ax.set_title(f"channel {channel_id}")
        ax.set_xlabel("run time")
        ax.tick_params("x", rotation=30)
        ax.set_ylabel("dG / %")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
    pdf.close()



    
    pdf_path = f"figures/calibration_summaries/dG_hist_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_path)
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        if channel_id in known_broken_channels[season][station_id]:
            continue
        ax.hist(dG[channel_id]*100,
                histtype="stepfilled",
                facecolor=colors_face[0],
                edgecolor=colors_edge[0],
                lw=3.)
        ax.set_yscale("log")
        ax.set_xlabel("dG / %")
        ax.set_ylabel("N")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
    pdf.close()
