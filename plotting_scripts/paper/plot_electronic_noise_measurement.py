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
    # MEASUREMENTS
    measurement_types = ["downhole", "surface"]
    measurement_types_fancy = ["downhole RF chain\n" + r"IGLU@-40$^\circ$ C", "surface RF chain\n"+ r"SURFACE@0$^\circ$ C"]
    electronic_noise_measurements_paths = ["electronic_noise_measurements/electronic_noise_digitized_downhole.json", "electronic_noise_measurements/electronic_noise_digitized_surface.json"]
    electronic_noise_measurements = {}
    for i, path in enumerate(electronic_noise_measurements_paths):
        with open(path, "r") as f:
            electronic_noise_dicts = json.load(f)

        electronic_noise_freq = np.array([float(d["x"]) * units.MHz for d in electronic_noise_dicts])
        electronic_noise = [float(d["y"]) for d in electronic_noise_dicts]

        electronic_noise_measurements[measurement_types[i]] = [electronic_noise_freq, electronic_noise]


    # PLOTTING
    plt.style.use("astroparticle_physics")


    fig, ax = plt.subplots()
    for i, electronic_noise_type in enumerate(measurement_types):
        electronic_noise_freq, electronic_noise = electronic_noise_measurements[electronic_noise_type]
        ax.plot(electronic_noise_freq / units.MHz, electronic_noise, label=measurement_types_fancy[i])

#    ax.set_ylim(25, 150)
    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("Electronic Noise Temp / K")
    ax.legend()
    fig.savefig("figures/paper/electronic_noise_measurements.eps", dpi=300)
    plt.close(fig)

