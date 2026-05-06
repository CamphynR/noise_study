import argparse
import os
import logging
import glob
import natsort
import numpy as np
from scipy import signal
from matplotlib import colormaps
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

    args.pickles = natsort.natsorted(args.pickles)
    
    # Files used for different antenna models were stored here
    # /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/antenna_model_variations

    n = np.array([1.3, 1.4, 1.5, 1.6, 1.7, 1.74])
    
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

    cmap_name = "viridis"
    cmap = colormaps[cmap_name].resampled(len(args.pickles)) 
    cmap_im = plt.imshow(n[np.newaxis, :], cmap=cmap)
    plt.colorbar(cmap_im)
    plt.savefig("figures/test")
    plt.clf()

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
            if "174" not in args.pickles[i]:
                ls = "solid"
                label = f"n = {n[i]}"
                lw = 1.
                alpha = 0.7
            else:
                ls = "solid"
                label = f"n = 1.74"
                lw = 1.5
                alpha = 1
            ax.plot(freqs, np.abs(spectrum[channel_id]), label=label,
#                    color=cmap_im.cmap((n[i] - n[0])/(n[-1] - n[0])),
                     lw=lw, ls=ls, alpha=alpha)
    ax.legend(loc="lower right", fontsize=12, ncols=3)
#    cbar = fig.colorbar(cmap_im, ax=ax)
#    cbar.ax.set_ylabel("index of refraction")
    ax.set_xlim(0,1.)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    fig.tight_layout()
    fig.savefig(f"figures/spectra_different_antenna.png", dpi=300, bbox_inches="tight")
