import argparse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities.utility_functions import convert_to_db, convert_error_to_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--channel_type", default="deep", choices=["deep", "hpol", "shallow"])
    args = parser.parse_args()

    station_ids = [11, 12, 13, 21, 22, 23, 24]
    season = args.season
    channel_type = args.channel_type
    gains = []
    for station_id in station_ids:
        radiant_calibration_path = f"absolute_amplitude_results/absolute_amplitude_calibration_season{season}_st{station_id}.csv"
        calibration_parameters = pd.read_csv(radiant_calibration_path)
        gain = calibration_parameters["gain"].to_numpy()
        el_ampl = calibration_parameters["el_ampl"].to_numpy()
        el_cst = calibration_parameters["el_cst"].to_numpy()
        f0 = calibration_parameters["f0"].to_numpy()
        gains.append(gain)

    gains = np.array(gains)
    gains = convert_to_db(gains)
    gains = gains.T


    channel_id_types = {"deep" :[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
                        "shallow" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    channel_subsets =  {"Phased Array" : [0, 1, 2, 3], "HPol" : [4, 8, 11, 21], "Helper VPol" : [5, 6, 7, 9, 10, 22, 23]}
#                        "Shallow Up" : [13, 16, 19],
#                        "Shalllow Down" : [12, 14, 15, 17, 18, 20]}
    channel_subsets = {i : key for key, value in zip(channel_subsets.keys(), channel_subsets.values()) for i in value}
    shape_subsets =  {"o" : [0, 1, 2, 3], "P" : [4, 8, 11, 21], "v" : [5, 6, 7, 9, 10, 22, 23]}
#                        "s" : [13, 16, 19],
#                        "d" : [12, 14, 15, 17, 18, 20]}
    shape_subsets = {i : key for key, value in zip(shape_subsets.keys(), shape_subsets.values()) for i in value}

    plt.style.use("astroparticle_physics")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, ax = plt.subplots()
    ax.violinplot(gains[channel_id_types[channel_type]], positions=None, vert=True, showmeans=True)
    for channel_id in channel_id_types[channel_type]:
        ax.scatter(np.arange(1, len(station_ids)+1), gains.T[:, channel_id],
                   color=colors[0],
                   marker=shape_subsets[channel_id],
                   facecolors="none")
    ax.set_xticks(np.arange(1, len(station_ids)+1), station_ids)
    ax.set_xlabel("Stations")
    ax.set_ylabel("Gains / dB")
#    ax.set_title(f"Overview of gain calibration season {season}, {channel_type} channels")
    legend_elements = [Line2D([0], [0], color=colors[0], lw=0, marker=shape, markerfacecolor="none") for shape in pd.unique(list(shape_subsets.values()))]
    ax.legend(legend_elements, pd.unique(list(channel_subsets.values())))
    fig.tight_layout()
    fig.savefig(f"figures/paper/gain_season{season}_channels_{channel_type}.pdf", dpi=300, bbox_inches="tight")
    
