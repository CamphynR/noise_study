import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
from NuRadioReco.utilities import units





if __name__ == "__main__":

    # DISCLAIMER WE DISTINGUISH BETWEEN OLD AND NEW DAQS, ONLY APPLICABLE FOR 2024

    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    known_broken_channels["2023"]["12"] = np.arange(24)
    known_broken_channels["2023"]["22"] = np.arange(24)


    seasons = [2022, 2023, "2024_radiant_v2"]

    seasons_int = []
    for season in seasons:
        if season == "2024_radiant_v2":
            season_int = 2024
        else:
            season_int = int(season)
        seasons_int.append(season_int)

    # SETTINGS
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    stations_new_daq = [11, 12, 13, 23]
    stations_old_daq = [21, 22, 24]

    vpols = [0, 1, 2, 3, 5, 6, 7, 9, 10, 19, 22, 23]
    hpols = [4, 8, 11, 21]
    lpda_up = [13, 16, 19]
    lpda_down = [12, 14, 15, 17, 18, 20]

    plt.style.use("astroparticle_physics")
    colors_style = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    antenna_colors = [colors_style[0], colors_style[2], colors_style[4], colors_style[3]]
    antenna_type_names = ["vpols", "hpols", "lpda up", "lpda down"]

    marker_styles = ["o", "s", "p", "x", "*", "8", "2", "^", "D", ".", "v"]
    color_styles = ["green", "blue", "red", "orange", "cyan", "gold", "black", "purple", "gray", "navy"]
    markers = {}
    colors = {}
    style_index = 0



    # READ CALIBRATION DATA
    calibration_per_season = []
    for season in seasons:
        calibration_result_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}"
        calibration_per_station = []
        for station_id in station_ids:
            if season == "2024_radiant_v2" and station_id in [21, 22]:
                calibration_per_station.append(-1)
                continue
            calibration_result_path = os.path.join(calibration_result_dir,
                                                   f"station{station_id}",
                                                   "default",
                    f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")

            calibration = pd.read_csv(calibration_result_path)
            calibration_per_station.append(calibration)
        calibration_per_season.append(calibration_per_station)


    # INVESTIGATE TEMPLATE USAGE PER STATION 
    templates_used_over_all_stations = []
    for season in seasons:
        for station_index, station_id in enumerate(station_ids):
            if season == "2024_radiant_v2" and station_id in [21, 22]:
                continue
            calibration = calibration_per_season[seasons.index(season)][station_ids.index(station_id)]
            templates = calibration["best_fit_template"].to_numpy()
            channel_ids = np.arange(len(templates))

            templates_used, nr_per_templates_used = np.unique(templates, return_counts=True)
            nr_templates_used = len(templates_used)

            for i, template in enumerate(templates_used):
                if template not in templates_used_over_all_stations:
                    templates_used_over_all_stations.append(template)
                    markers[template] = marker_styles[style_index]
                    colors[template] = color_styles[style_index]
                    style_index += 1
                mask = templates == template

    # SUMMARIZE OVER ALL STATIONS
    # we distinguish between stations with a new and old DAQ (in 2024!)

    
    figname = f"figures/overviews/response_template_summary.pdf"
    pdf = PdfPages(figname)

    fig, ax = plt.subplots()
    max_template_count = 0
    for template in templates_used_over_all_stations:

        template_counts = np.zeros_like(channel_ids)
        for season in seasons:
            for station_id in station_ids:
                if season == "2024_radiant_v2" and station_id in [21, 22]:
                    continue
                calibration = calibration_per_season[seasons.index(season)][station_ids.index(station_id)]
                for channel_id in channel_ids:
                    if channel_id in known_broken_channels[str(season)][str(station_id)]:
                        continue
                
                    template_used = template == calibration["best_fit_template"][channel_id]
                    if calibration["best_fit_template"][channel_id] == "v2_ch11":
                        print(season)
                        print(station_id)
                        print(channel_id)
                        
                    template_counts[channel_id] += template_used
        for channel_id in channel_ids:
            if template_counts[channel_id] > max_template_count:
                max_template_count = template_counts[channel_id]
        mask = np.nonzero(template_counts)
        ax.scatter(channel_ids[mask], template_counts[mask], s=80,
                   marker=markers[template], color=colors[template],
                   label = f"{template}")


    ax.set_xticks(channel_ids)
    ax.set_xticklabels(channel_ids, rotation=-90, size=12)
    antenna_legend_elements = []
    for j, antenna_type in enumerate([vpols, hpols, lpda_up, lpda_down]):
        ax.bar(antenna_type, max_template_count * np.ones_like(antenna_type),
               width=1.,
               color=antenna_colors[j],
               alpha=0.4,
               zorder=-1)
        antenna_legend_elements.append(Patch(facecolor=antenna_colors[j],
                                             label=antenna_type_names[j]))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Channel")
    ax.set_ylabel("counts")
    antenna_type_legend = fig.legend(handles=antenna_legend_elements,
                      loc="lower center", bbox_to_anchor=(0.5, 1.01),
                                     ncols=4)
    ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
    ax.add_artist(antenna_type_legend)
    fig.tight_layout()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)



    pdf.close()
