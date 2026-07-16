import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd






if __name__ == "__main__":
    seasons = ["2022", "2023", "2024_radiant_v2"]
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    channel_ids = [4, 8, 11, 21]

    gof_hpol_v4 = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    gof_hpol_v5 = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    for season_idx, season in enumerate(seasons):
        for station_idx, station_id in enumerate(station_ids):
            if season == "2024_radiant_v2" and station_id in [21, 22]:
                continue
            if season == "2022" and station_id in [22]:
                continue
            if season == "2023" and station_id in [12, 22]:
                continue

            hpol_v4_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default_hpol_v4"
            hpol_v5_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default"

            hpol_v5_calibration = pd.read_csv(
                    os.path.join(hpol_v5_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"),
                    index_col=0)
            for channel_idx, channel_id in enumerate(channel_ids):
                gof_hpol_v5[season_idx, station_idx, channel_idx] = hpol_v5_calibration["gof_value"][channel_id]
                best_fit_template_hpol_v5 = hpol_v5_calibration["best_fit_template"][channel_id]
                if season=="2024_radiant_v2" and station_id == 13:
                    hpol_v4_calibration = pd.read_csv(
                            os.path.join(hpol_v4_calibration_path,
                                     f"absolute_amplitude_calibration_season{season}_st{station_id}_default_hpol_v4_key{best_fit_template_hpol_v5}.csv")
                            )
                else:
                    hpol_v4_calibration = pd.read_csv(
                            os.path.join(hpol_v4_calibration_path,
                                     f"absolute_amplitude_calibration_season{season}_st{station_id}_key{best_fit_template_hpol_v5}.csv")
                            )
                gof_hpol_v4[season_idx, station_idx, channel_idx] = hpol_v4_calibration["gof"][channel_id]
                if hpol_v4_calibration["gof"][channel_id] < 0.01:
                    print(season)
                    print(station_id)
                    print(channel_id)
                    print(best_fit_template_hpol_v5)
                    print(hpol_v4_calibration["gof"])



    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()

    ax.hist(np.ndarray.flatten(gof_hpol_v5),
            edgecolor=colors[0],
            facecolor=colors[0] + "88",
            lw = 2.5,
            label= "hpol v5",
            histtype="stepfilled")

    ax.hist(np.ndarray.flatten(gof_hpol_v4),
            edgecolor=colors[1],
            facecolor=colors[1] + "88",
            lw = 2.5,
            label= "hpol v4",
            histtype="stepfilled")

    ax.set_xlabel("gof (chi2)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/antenna_tests/gof_hpol_v4_vs_hpol_v5")

