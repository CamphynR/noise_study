import argparse
import os
import logging
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from NuRadioReco.utilities import fft
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader

from utilities.utility_functions import read_config, read_pickle




def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"], contents["header"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-p", "--pickles", nargs = "+")
    parser.add_argument("--plot_range", action="store_true")
    args = parser.parse_args()
    
    # Files used for different antenna models were stored here
    # /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/antenna_model_variations
    
    spectra = []
    var_spectra = []
    headers = []
    antenna_models = []
    for pickle in args.pickles:
        freqs, spectrum, var_spectrum, header = read_average_freq_spectrum_from_pickle(pickle)
        spectra.append(spectrum)
        var_spectra.append(var_spectrum)
        headers.append(header)
        config_path = pickle.rsplit("/", 2)[0] + "/config.json"
        config_temp = read_config(config_path)
        antenna_models.append(config_temp["antenna_models"])

    spectra = np.array(spectra)
    var_spectra = np.array(var_spectra)



    plt.style.use("astroparticle_physics")
    channel_id = 0
    channel_mapping = {key : "VPol" for key in [0, 1, 2, 3, 5, 6, 7 , 9, 10, 22, 23]}
    for key in [4, 8, 11, 21]:
        channel_mapping[key] = "HPol"
    for key in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        channel_mapping[key] = "LPDA"
    fig, ax = plt.subplots()
    if args.plot_range:
        spectra_tmp = spectra[:,channel_id].T
        spectra_minima = [np.min(spectra_at_freq) for spectra_at_freq in spectra_tmp]
        spectra_maxima = [np.max(spectra_at_freq) for spectra_at_freq in spectra_tmp]
        plt.fill_between(freqs, spectra_minima, spectra_maxima)
    else:
        for i, spectrum in enumerate(spectra):
            ax.plot(freqs, np.abs(spectrum[channel_id]), label = f"spectrum {antenna_models[i][channel_mapping[channel_id]]}")
#    ax.legend(loc = "lower center", fontsize=12)
    ax.set_xlim(0,1.)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    fig.tight_layout()
    fig.savefig(f"figures/spectra_different_antenna.png", dpi=300)
