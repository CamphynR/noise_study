from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle








if __name__ == "__main__":
    result_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results"

    seasons = [2023]
    station_ids = [11]

    for season in seasons:
        for station_id in station_ids:
            result_path = os.path.join(result_dir,
                                         f"season{season}",
                                         f"station{station_id}",
                                         "slope_per_run",
                                         f"season{season}_st{station_id}_all_runs_compiled_slope_per_run.pickle"
                                         )
                                    
            with open(result_path, "rb") as file:
                calibration_per_run = pickle.load(file)

            run_nrs = calibration_per_run["run_nr"]
            gain_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in calibration_per_run["calibration"]])
            gain_error_per_run = np.array([[cal_ch["gain"].error for cal_ch in cal_run["fit_results"]] for cal_run in calibration_per_run["calibration"]])
            slope_per_run = np.array([[cal_ch["slope"].value for cal_ch in cal_run["fit_results"]] for cal_run in calibration_per_run["calibration"]])
            slope_error_per_run = np.array([[cal_ch["slope"].error for cal_ch in cal_run["fit_results"]] for cal_run in calibration_per_run["calibration"]])
             


    antenna_types = {
        "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
        "HPol" : [4, 8, 11, 21],
        "LPDA up" : [13, 16, 19],
        "LPDA down" : [12, 14, 15, 17, 18, 20],
        }

    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        for i, channel_id in enumerate(channel_ids):
            axs[ax_i].errorbar(run_nrs, slope_per_run[:, channel_id],
                               yerr = slope_error_per_run[:, channel_id],
                               fmt="o",
                               markersize=0.2,
                              color=colors[i])
        axs[ax_i].set_title(channel_type)

    fig.tight_layout()
    fig.text(0.5, 0., "run number", ha="center")
    fig.text(0., 0.5, "slope value", ha="center", va="center", rotation=90)
    fig.savefig("figures/tests/slope_over_time", bbox_inches="tight")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        for i, channel_id in enumerate(channel_ids):
            axs[ax_i].hist(slope_per_run[:, channel_id],
                              facecolor=(0, 0, 0, 0),
                              histtype="stepfilled",
                              edgecolor=colors[i],
                              label=channel_id)
        axs[ax_i].set_title(channel_type)
    axs[2].legend()
    axs[3].legend()

    fig.text(0.5, 0., "slope value / 1/GHz", ha="center")
    fig.tight_layout()
    fig.savefig("figures/tests/slope_over_time_hist")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        for i, channel_id in enumerate(channel_ids):
            axs[ax_i].errorbar(run_nrs, gain_per_run[:, channel_id],
                               yerr = gain_error_per_run[:, channel_id],
                               ls=None,
                               marker=".",
                               markersize=0.2,
                              color=colors[i])

    fig.savefig("figures/tests/gain_over_time", bbox_inches="tight")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        for i, channel_id in enumerate(channel_ids):
            axs[ax_i].hist(gain_per_run[:, channel_id],
                              facecolor=(0, 0, 0, 0),
                              histtype="stepfilled",
                              edgecolor=colors[i],
                              label=channel_id)

        axs[ax_i].set_title(channel_type)
    fig.savefig("figures/tests/gain_over_time_hist")



    pdf = PdfPages("figures/tests/slope_gain_over_time.pdf")

    channel_ids = np.arange(24) 
    for channel_id in channel_ids:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].errorbar(run_nrs, slope_per_run[:, channel_id],
                           yerr = slope_error_per_run[:, channel_id],
                           fmt="o",
                           markersize=1.,
                          color=colors[0])
        axs[0].set_ylabel("slope value")
        axs[0].set_title(channel_id)


        axs[1].errorbar(run_nrs, gain_per_run[:, channel_id],
                           yerr = gain_error_per_run[:, channel_id],
                           fmt="o",
                           markersize=1.,
                          color=colors[1])

        axs[1].set_xlabel("run number")
        axs[1].set_ylabel("gain")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)

    pdf.close()

