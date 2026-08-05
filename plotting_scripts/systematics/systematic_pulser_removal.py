import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd






if __name__ == "__main__":
    seasons = ["2023"]
    station_ids = [11, 13, 21, 23, 24]
    channel_ids = np.arange(24)

    gof_pulser = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    gof_default = np.zeros((len(seasons), len(station_ids), len(channel_ids)))

    gain_pulser = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    gain_default = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    for season_idx, season in enumerate(seasons):
        for station_idx, station_id in enumerate(station_ids):
            if season == "2024_radiant_v2" and station_id in [21, 22]:
                continue
            if season == "2022" and station_id in [22]:
                continue
            if season == "2023" and station_id in [12, 22]:
                continue

            pulser_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/test_pulser_remove"
            default_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default"

            default_calibration = pd.read_csv(
                    os.path.join(default_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"),
                    index_col=0)
            for channel_idx, channel_id in enumerate(channel_ids):
                gof_default[season_idx, station_idx, channel_idx] = default_calibration["gof_value"][channel_id]
                gain_default[season_idx, station_idx, channel_idx] = default_calibration["gain"][channel_id]
                best_fit_template_default = default_calibration["best_fit_template"][channel_id]
                pulser_calibration = pd.read_csv(
                        os.path.join(pulser_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_test_pulser_remove_key{best_fit_template_default}.csv")
                        )
                gof_pulser[season_idx, station_idx, channel_idx] = pulser_calibration["gof"][channel_id]
                gain_pulser[season_idx, station_idx, channel_idx] = pulser_calibration["gain"][channel_id]


    antenna_types = {
            "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA up" : [13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20]
            }


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    

    
    fig, axs = plt.subplots(2, 2, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)

    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        axs[ax_i].hist(np.ndarray.flatten(gof_default[:, :, channel_ids]),
                edgecolor=colors[0],
                facecolor=colors[0] + "88",
                lw = 2.5,
                label= "default",
                histtype="stepfilled")

        axs[ax_i].hist(np.ndarray.flatten(gof_pulser[:, :, channel_ids]),
                edgecolor=colors[1],
                facecolor=colors[1] + "88",
                lw = 2.5,
                label= "pulser removed",
                histtype="stepfilled")

        axs[ax_i].set_title(channel_type)

    for ax in axs:
        ax.set_xlabel("gof (chi2)")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/tests/gof_default_vs_pulser_removed")

    fig, axs = plt.subplots(2, 2, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)

    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        axs[ax_i].hist(np.ndarray.flatten(gain_default[:, :, channel_ids]),
                edgecolor=colors[0],
                facecolor=colors[0] + "88",
                lw = 2.5,
                label= "default",
                histtype="stepfilled")

        axs[ax_i].hist(np.ndarray.flatten(gain_pulser[:, :, channel_ids]),
                edgecolor=colors[1],
                facecolor=colors[1] + "88",
                lw = 2.5,
                label= "pulser removed",
                histtype="stepfilled")

        axs[ax_i].set_title(channel_type)

    for ax in axs:
        ax.set_xlabel("gain")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/tests/gain_default_vs_pulser_removed")



    fig, axs = plt.subplots(2, 2, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)

    for ax_i, (channel_type, channel_ids) in enumerate(antenna_types.items()):
        axs[ax_i].hist(100* (np.ndarray.flatten(gain_default[:, :, channel_ids]) - np.ndarray.flatten(gain_pulser[:, :, channel_ids]))/np.ndarray.flatten(gain_default[:, :, channel_ids]),
                edgecolor=colors[0],
                facecolor=colors[0] + "88",
                lw = 2.5,
                label= "(default - pulser removed) / default",
                histtype="stepfilled")


        axs[ax_i].set_title(channel_type)
    if ax_i == len(antenna_types)-1:
        handles, labels = axs[ax_i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncols=5, fontsize="large", bbox_to_anchor=(0.5, 1.005))

    for ax in axs:
        ax.set_xlabel("dG / %")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig("figures/tests/gain_relative_default_vs_pulser_removed")
