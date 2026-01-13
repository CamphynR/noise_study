import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd
from NuRadioReco.utilities import units





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args()

    # DISCLAIMER WE DISTINGUISH BETWEEN OLD AND NEW DAQS, ONLY APPLICABLE FOR 2024

    # SETTINGS
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    stations_new_daq = [11, 12, 13, 23]
    stations_old_daq = [21, 22, 24]

    vpols = [0, 1, 2, 3, 5, 6, 7, 9, 10, 19, 22, 23]
    hpols = [4, 8, 11, 21]
    lpda_up = [13, 16, 19]
    lpda_down = [12, 14, 15, 17, 18, 20]
    antenna_colors = ["pink", "blue", "indigo", "gray"]
    antenna_type_names = ["vpols", "hpols", "lpda up", "lpda down"]

    plt.style.use("retro")
    marker_styles = ["o", "s", "p", "x", "*", "8", "2", "^", "D", ".", "v"]
    color_styles = ["green", "blue", "red", "orange", "cyan", "gold", "black", "purple", "gray", "navy"]
    markers = {}
    colors = {}
    style_index = 0

    figname = f"figures/calibrated_gain_summaries/response_template_summary_season{args.season}.pdf"
    pdf = PdfPages(figname)


    # READ CALIBRATION DATA
    calibration_result_dir = "/user/rcamphyn/noise_study/absolute_amplitude_results"
    calibration_per_station = []
    for station_id in station_ids:
        calibration_result_path = os.path.join(calibration_result_dir,
                f"absolute_amplitude_calibration_season{args.season}_st{station_id}_best_fit.csv")

        calibration = pd.read_csv(calibration_result_path)
        calibration_per_station.append(calibration)


    # INVESTIGATE TEMPLATE USAGE PER STATION 
    templates_used_over_all_stations = []
    for station_index, station_id in enumerate(station_ids):
        calibration = calibration_per_station[station_index]
        templates = calibration["best_fit_template"].to_numpy()
        gof_values = calibration["gof_value"].to_numpy()

        # we assume the same goodness of fit function was used for all the channels
        # (otherwise a comparison doesn't make sense anyway)
        gof_method = calibration["gof_method"][0]
        channel_ids = np.arange(len(templates))

        templates_used, nr_per_templates_used = np.unique(templates, return_counts=True)
        nr_templates_used = len(templates_used)

        fig, ax = plt.subplots()
        for i, template in enumerate(templates_used):
            if template not in templates_used_over_all_stations:
                templates_used_over_all_stations.append(template)
                markers[template] = marker_styles[style_index]
                colors[template] = color_styles[style_index]
                style_index += 1
            mask = templates == template
            ax.scatter(channel_ids[mask], gof_values[mask],
                       s=80, marker=markers[template], color=colors[template],
                       label=f"{template}\ncounts: {nr_per_templates_used[i]}")
        ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
        ax_title = f"station {station_id}"
        if station_id in stations_old_daq:
            ax_title += " (old DAQ)"
        ax.set_title(ax_title)
        ax.set_xticks(channel_ids)
        ax.set_xlabel("Channel")
        ax.set_ylabel(gof_method)
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)

    # SUMMARIZE OVER ALL STATIONS
    # we distinguish between stations with a new and old DAQ (in 2024!)
    
    fig, (ax_old_daq, ax_new_daq) = plt.subplots(1, 2)
    for template in templates_used_over_all_stations:

        template_counts = np.zeros_like(channel_ids)
        for station_id in stations_old_daq:
            calibration = calibration_per_station[station_ids.index(station_id)]
            templates = calibration["best_fit_template"].to_numpy()
            template_counts += template == templates
        ax_old_daq.scatter(channel_ids, template_counts, s=80,
                   marker=markers[template], color=colors[template],
                   label = f"{template}")

        template_counts = np.zeros_like(channel_ids)
        for station_id in stations_new_daq:
            calibration = calibration_per_station[station_ids.index(station_id)]
            templates = calibration["best_fit_template"].to_numpy()
            template_counts += template == templates
        ax_new_daq.scatter(channel_ids, template_counts, s=80,
                   marker=markers[template], color=colors[template],
                   label = f"{template}")

    ax_old_daq.set_title("old DAQ")
    ax_new_daq.set_title("new DAQ")
    for ax in (ax_old_daq, ax_new_daq):
        ax.set_xticks(channel_ids)
        ax.set_xticklabels(channel_ids, rotation=-90, size=12)
        antenna_legend_elements = []
        for j, antenna_type in enumerate([vpols, hpols, lpda_up, lpda_down]):
            ax.bar(antenna_type, 4 * np.ones_like(antenna_type),
                   width=1.,
                   color=antenna_colors[j],
                   alpha=0.4,
                   zorder=-1)
            antenna_legend_elements.append(Patch(facecolor=antenna_colors[j],
                                                 label=antenna_type_names[j]))
        ax.set_xlabel("Channel")
        ax.set_ylabel("# uses")
    antenna_type_legend = plt.legend(handles=antenna_legend_elements,
                      loc="lower left", bbox_to_anchor=(1., 0.))
    ax_new_daq.legend(loc="upper left", bbox_to_anchor=(1., 1.))
    ax.add_artist(antenna_type_legend)
    fig.tight_layout()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)







    pdf.close()
