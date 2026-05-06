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
    parser.add_argument("--station", type=int)
    args = parser.parse_args()
    season = 2023
    station_id = args.station
    
    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    # define the groups for which to calculate an uncertainty
    channel_groups = {"VPols" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                      "HPol" : [4, 8, 11, 21],
                      "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}
    
    channel_groups_tmp = copy.deepcopy(channel_groups)
    for key, ch_list in channel_groups_tmp.items():
        for ch_id in ch_list:
            if ch_id in known_broken_channels[str(season)][str(args.station)]:
                channel_groups[key].remove(ch_id)


    

    fit_ranges = [
            "fit_range_150 600",
            "fit_range_50 600",
            "fit_range_150 700",
            "fit_range_50 700",
#            "fit_range_250_700",
                  ]

    plot_data = {}
    for fit_range in fit_ranges:
        plot_data_path = f"absolute_amplitude_results/season{season}/station{station_id}/{fit_range}/absolute_amplitude_calibration_season{season}_st{station_id}_{fit_range}_plot_data.pickle"
        with open(plot_data_path, "rb") as plot_data_file:
            plot_data[fit_range] = pickle.load(plot_data_file)
          
    channel_ids = plot_data[fit_ranges[0]]["channel_ids"]
    frequencies = plot_data[fit_ranges[0]]["frequencies"]
    data = plot_data[fit_ranges[0]]["data"]


    calibration = {}
    for fit_range in fit_ranges:
        calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/{fit_range}"
        calibration_file = os.path.join(calibration_dir,
                                        f"absolute_amplitude_calibration_season{season}_st{station_id}_{fit_range}_best_fit.csv")
                             
        calibration[fit_range] = pd.read_csv(calibration_file)



    spread = [cal["gain"].values for cal in calibration.values()] 
    spread = np.array(spread)
    spread_dB = np.std(convert_to_db(spread), axis=0)

    



    plt.style.use("retro")
    pdf_path = f"figures/fitting_tests/effect_fit_range_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_path)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.plot(frequencies, data[channel_id], label="data")
        for fit_range in fit_ranges:
            ax.plot(frequencies, plot_data[fit_range]["sim"][channel_id],
                    label=f"sim {fit_range}")

        ax.legend()
        ax.set_xlim(0, 1.)
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()


    plt.style.use("retro")
    pdf_path = f"figures/fitting_tests/effect_fit_range_season{season}_st{station_id}_cal_params.pdf"
    pdf = PdfPages(pdf_path)

    fig, axs = plt.subplots(3, 1, sharex=True)
    for fit_range in fit_ranges:
        axs[0].scatter(channel_ids,
                   convert_to_db(calibration[fit_range]["gain"]),
                   marker=10,
                   label=f"{fit_range}")
        axs[1].scatter(channel_ids,
                       calibration[fit_range]["gof_value"])

        axs[2].scatter(channel_ids,
                       spread_dB)

    axs[0].legend(loc="upper left", bbox_to_anchor=(1., 1.))
    axs[0].set_ylabel("Gain / dB")
    axs[1].set_ylabel("gof / reduced chi2")
    axs[2].set_xlabel("channels")
    axs[2].set_ylabel("Spread on gain / dB")
    axs[2].set_xticks(channel_ids)

    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()




    # DEFINING SYSTEMATICS PER ANTENNA TYPE
    # =====================================


    pdf_path = f"figures/fitting_tests/effect_fit_range_season{season}_st{station_id}_systematics.pdf"
    pdf = PdfPages(pdf_path)

    for antenna_type in channel_groups.keys():
        fig, ax = plt.subplots()
        for channel_id in channel_groups[antenna_type]:
            differences = [] 
            i = 1
            for fit_range in fit_ranges:
                gain_db = convert_to_db(calibration[fit_range]["gain"][channel_id])
                for next_fit_range in fit_ranges[i:]:
                    next_gain_db = convert_to_db(calibration[next_fit_range]["gain"][channel_id])
                    differences.append(np.abs(gain_db - next_gain_db))
                i += 1
        ax.hist(differences)

        fig.suptitle(antenna_type)
        fig.savefig(pdf, format="pdf")

    pdf.close()
