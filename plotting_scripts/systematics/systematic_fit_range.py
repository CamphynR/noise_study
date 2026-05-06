import argparse
import copy
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


from utilities.utility_functions import convert_to_db




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--station", nargs="+", type=int)
    args = parser.parse_args()

    season = args.season
    station_ids = args.station
    
    # define the groups for which to calculate an uncertainty
    channel_groups = {"VPols" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                      "HPol" : [4, 8, 11, 21],
                      "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}


    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    
    channel_groups_stations = []

    for station_id in station_ids:
        channel_groups_tmp = copy.deepcopy(channel_groups)
        for key, ch_list in channel_groups.items():
            for ch_id in ch_list:
                if ch_id in known_broken_channels[str(season)][str(station_id)]:
                    channel_groups_tmp[key].remove(ch_id)
        channel_groups_stations.append(channel_groups_tmp)

    

    fit_ranges = [
            "fit_range_150 600",
            "fit_range_50 600",
            "fit_range_150 700",
            "fit_range_50 700",
#            "fit_range_250_700",
                  ]


    data_stations = []
    calibration_stations = []
    for station_id in station_ids:
        plot_data = {}
        for fit_range in fit_ranges:
            plot_data_path = f"absolute_amplitude_results/season{season}/station{station_id}/{fit_range}/absolute_amplitude_calibration_season{season}_st{station_id}_{fit_range}_plot_data.pickle"
            with open(plot_data_path, "rb") as plot_data_file:
                plot_data[fit_range] = pickle.load(plot_data_file)
              
        channel_ids = plot_data[fit_ranges[0]]["channel_ids"]
        frequencies = plot_data[fit_ranges[0]]["frequencies"]
        data = plot_data[fit_ranges[0]]["data"]
        data_stations.append(data)


        calibration = {}
        for fit_range in fit_ranges:
            calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/{fit_range}"
            calibration_file = os.path.join(calibration_dir,
                                            f"absolute_amplitude_calibration_season{season}_st{station_id}_{fit_range}_best_fit.csv")
                                 
            calibration[fit_range] = pd.read_csv(calibration_file)

        calibration_stations.append(calibration)



    


    plt.style.use("retro")
    pdf_path = f"figures/fitting_tests/effect_fit_range_season{season}_all_stations_cal_params.pdf"
    pdf = PdfPages(pdf_path)

    fig, axs = plt.subplots(2, 1, sharex=True)
    for fit_range in fit_ranges:
        for calibration in calibration_stations:
            axs[0].scatter(channel_ids,
                       convert_to_db(calibration[fit_range]["gain"]),
                       marker=10,
                       label=f"{fit_range}")
            axs[1].scatter(channel_ids,
                           calibration[fit_range]["gof_value"])


    axs[0].legend(loc="upper left", bbox_to_anchor=(1., 1.))
    axs[0].set_ylabel("Gain / dB")
    axs[1].set_ylabel("gof / reduced chi2")
    axs[1].set_xlabel("channels")
    axs[1].set_xticks(channel_ids)

    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()




    # DEFINING SYSTEMATICS PER ANTENNA TYPE
    # =====================================


    pdf_path = f"figures/fitting_tests/effect_fit_range_season{season}_all_stations_systematics.pdf"
    pdf = PdfPages(pdf_path)

    for antenna_type in channel_groups.keys():
        fig, ax = plt.subplots()

        differences_id = []
        differences = [] 
        for station_idx, station_id in enumerate(station_ids):
            for channel_id in channel_groups_stations[station_idx][antenna_type]:
                i = 1
                for fit_range in fit_ranges:
                    gain_db = convert_to_db(calibration_stations[station_idx][fit_range]["gain"][channel_id])
                    for next_fit_range in fit_ranges[i:]:
                        next_gain_db = convert_to_db(calibration_stations[station_idx][next_fit_range]["gain"][channel_id])
                        differences.append(np.abs(gain_db - next_gain_db))
                        differences_id.append({"station" : station_id, "channel" : channel_id, "fit_ranges" : [fit_range, next_fit_range]})
                    i += 1
        ax.hist(differences)
        max_difference = np.max(differences)
        max_difference_id = differences_id[np.where(differences == max_difference)[0][0]]
        ax.text(0.95, 0.95,
                f'max dG = {max_difference:.2E} dB\nstation {max_difference_id["station"]}\nchannel {max_difference_id["channel"]}\nbetween {max_difference_id["fit_ranges"][0]} and {max_difference_id["fit_ranges"][1]}',
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox={"boxstyle" : "round",
                      "color" : "white"})
        ax.set_xlabel("dG / dB")
        ax.set_ylabel("counts")

        fig.suptitle(antenna_type)
        fig.savefig(pdf, format="pdf")

    pdf.close()
