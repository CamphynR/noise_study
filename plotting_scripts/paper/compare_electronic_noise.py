import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import constants

from NuRadioReco.utilities import units


def volt_to_temp(volt, min_freq, max_freq, frequencies, resistance=50*units.ohm, filter_type="rectangular"):
    if filter_type=="rectangular":
        filt = np.zeros_like(frequencies)
        filt[np.where(np.logical_and(min_freq < frequencies , frequencies < max_freq))] = 1
    bandwidth = np.trapezoid(np.abs(filt)**2, frequencies)
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)

    temp = volt**2 / (k * bandwidth * resistance)
    return temp


def electronic_noise_weight(freq, el_ampl, el_cst, f0):
    return el_ampl * (freq - f0) + el_cst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=24)
    args = parser.parse_args()

    season = 2024
    station_id = args.station
    nr_channels = 24

    channel_types = {"downhole" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
                     "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}
    channel_types = {str(key): value  for value, dict_list in zip(channel_types.keys(),channel_types.values()) for key in dict_list}
    channel_types_fancy = {"Downhole (IGLU at -40 C)" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
                           "Surface (SURFACE at 0 C)" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}
    channel_types_fancy = {str(key): value  for value, dict_list in zip(channel_types_fancy.keys(),channel_types_fancy.values()) for key in dict_list}

    electronic_noise_temp_sim = 80 * units.K


    # MEASUREMENTS
    measurement_types = ["downhole", "surface"]
    electronic_noise_measurements_paths = ["electronic_noise_measurements/electronic_noise_digitized_downhole.json", "electronic_noise_measurements/electronic_noise_digitized_surface.json"]
    electronic_noise_measurements = {}
    for i, path in enumerate(electronic_noise_measurements_paths):
        with open(path, "r") as f:
            electronic_noise_dicts = json.load(f)

        electronic_noise_freq = np.array([float(d["x"]) * units.MHz for d in electronic_noise_dicts])
        electronic_noise = [float(d["y"]) for d in electronic_noise_dicts]

        electronic_noise_measurements[measurement_types[i]] = [electronic_noise_freq, electronic_noise]


    # CALIBRATION

    calibration_dir = "absolute_amplitude_results"
    try:
        calibration_path = os.path.join(calibration_dir, f"absolute_amplitude_calibration_season{season}_st{station_id}.csv")
        calibration_ds = pd.read_csv(calibration_path)
    except:
        calibration_path = os.path.join(calibration_dir, f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")
        calibration_ds = pd.read_csv(calibration_path)


    electronic_noise_calibration = []
    for channel_id in range(nr_channels):
        electronic_noise_freq = electronic_noise_measurements[channel_types[str(channel_id)]][0]
        electronic_weight = [electronic_noise_weight(frequency,
                                                     calibration_ds["el_ampl"].to_numpy()[channel_id],
                                                     calibration_ds["el_cst"].to_numpy()[channel_id],
                                                     calibration_ds["f0"].to_numpy()[channel_id])
                             for frequency in electronic_noise_freq]

        electronic_weight = np.array(electronic_weight)
        electronic_noise_calibration.append(electronic_weight * electronic_noise_temp_sim)


    # PLOTTING
    plt.style.use("astroparticle_physics")


    pdf = PdfPages(f"figures/paper/electronic_noise_season{season}_st{station_id}.pdf")
    for channel_id in range(24):
        electronic_noise_freq, electronic_noise = electronic_noise_measurements[channel_types[str(channel_id)]]
        electronic_noise_cal = electronic_noise_calibration[channel_id]
        fig, ax = plt.subplots()
        ax.plot(electronic_noise_freq / units.MHz, electronic_noise, label=channel_types_fancy[str(channel_id)])
        ax.plot(electronic_noise_freq / units.MHz, electronic_noise_cal, label="Calibration")

    #    ax.set_ylim(25, 150)
        ax.set_xlabel("freq / MHz")
        ax.set_ylabel("Electronic Noise Temp / K")
#        ax.set_title(f"channel {channel_id}")
        ax.legend()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()




    # ALTERNATIVE PLOTTING
    colors = [d['color'] for d in plt.rcParams["axes.prop_cycle"]]

    phased_arrays = [0, 1, 2, 3]
    fig, ax = plt.subplots()
    data_line = ax.plot(electronic_noise_freq / units.MHz, electronic_noise, label=channel_types_fancy[str(channel_id)])
    for channel_id in phased_arrays:
        electronic_noise_cal = electronic_noise_calibration[channel_id]
        cal_line = ax.plot(electronic_noise_freq / units.MHz, electronic_noise_cal,
                            color=colors[1], label="Phased array calibration")


    ax.set_xlabel("frequency / MHz")
    ax.set_ylabel("electronic noise temp / K")
    ax.legend(handles=[data_line[0], cal_line[0]], loc="upper left")
    fig.savefig(f"figures/paper/electronic_noise_season{season}_st{station_id}_PA.pdf",
                bbox_inches="tight", dpi=300)
