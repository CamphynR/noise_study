"""
Main module used to simulate noise. NuRadio's formalism for noise is used, in which
the assumption that the amplitudes in the time domain are normal distributed leads to
a Rayleigh distribution in the energy spectral density - i.e. the Fourier transform formalism of NuRadio -
(TODO see notes in pdf)
"""
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import subprocess

from NuRadioReco.utilities import units
from NuRadioReco.detector.RNO_G.rnog_detector import Detector

from spectralAmplitudeHistogramSimulator import spectralAmplitudeHistogramSimulator
from utility_functions import open_config

def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = open_config(args.config)
    hist_range = [0., 0.07]
    nr_bins = 128
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)

    detector = Detector(select_stations=args.station)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    temp = 300 * units.kelvin

    # for now use antenna bandwidth as described in the rno-g paper
    bandwidth = 0.6 - 0.15 * units.GHz

    hist_simulator = spectralAmplitudeHistogramSimulator()
    hist_simulator.begin(args.station, detector, temp, bandwidth)

    savedir = "data/plotting_data/spec_hist_sim"
    try:
        os.makedirs(savedir)
    except OSError:
        pass
    savefile = f"{savedir}/spec_hist_s{args.station}_sim.pickle"
    frequencies, spec_hists = hist_simulator.simulate(args.samples, nr_bins, hist_range, savefile=savefile)



#    param, cov = curve_fit(rayleigh, bin_centers, spec_hist[0][300])
#    ray = rayleigh(bin_centers, param[0])
    plt.style.use("/user/rcamphyn/envs/gaudi.mplstyle")
#    fig, axs = plt.subplots(2, 1)
#    axs[0].stairs(spec_hist[0][300], edges = bin_edges)
#    axs[0].plot(bin_centers, rayleigh(bin_centers, param[0]))
#
#    axs[1].plot(bin_centers, (spec_hist[0][300]**2 - ray**2)/ray)

    channel = 0
    args.freq_lim = [0.1, 0.8]
    fig, ax = plt.subplots()
    freq_idxs = np.nonzero((args.freq_lim[0] < frequencies[channel]) & (frequencies[channel] < args.freq_lim[1]))[0]
    cmesh = ax.pcolormesh(bin_centers, frequencies[channel][freq_idxs], spec_hists[channel, freq_idxs])
    ax.set_title(f"Simulated spectral magnitude, station {args.station}, channel {channel}, with detector, {args.samples} samples")
    ax.set_xlabel("Spectral magnitude / V/GHz")
    ax.set_ylabel("frequencies / GHz")
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.set_label("counts")
    fig.tight_layout()
    fig.savefig("test.png")
