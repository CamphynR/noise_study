import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utilities.utility_functions import convert_to_db



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    args = parser.parse_args()

    station_ids = [11, 13, 21, 22, 23, 24]


    calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{args.season}"
    calibration_files = [os.path.join(calibration_dir,
                                      f"station{station_id}",
                                      "measured_noise_no_weight_new_impedance_cable_11",
                                      f"absolute_amplitude_calibration_season{args.season}_st{station_id}_measured_noise_no_weight_new_impedance_cable_11_best_fit.csv")
                         for station_id in station_ids]


    calibrations = [pd.read_csv(cal_file) for cal_file in calibration_files]

    plt.style.use("astroparticle_physics")
    pdf_path = f"figures/overviews/outliers_season{args.season}.pdf"
    pdf = PdfPages(pdf_path)

    channel_ids = np.arange(24)
    for i, station_id in enumerate(station_ids):
        calibration = calibrations[i]
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].scatter(channel_ids, convert_to_db(calibration["gain"]))
        axs[0].set_ylabel("gain / dB")

        axs[1].scatter(channel_ids, calibration["gof_value"]) 
        axs[1].set_ylabel(f"{calibration['gof_method'][0]} / a.u.")
        
        axs[1].set_xticks(channel_ids, labels=channel_ids, rotation=-45)
        axs[1].set_xlabel("channels")

        fig.suptitle(f"station {station_id}")
        fig.savefig(pdf, format="pdf")



    pdf.close()
