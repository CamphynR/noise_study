"""
Main module used to simulate noise. NuRadio's formalism for noise is used, in which
the assumption that the amplitudes in the time domain are normal distributed leads to
a Rayleigh distribution in the energy spectral density - i.e. the Fourier transform formalism of NuRadio -
(TODO see notes in pdf)
"""
import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

from NuRadioReco.utilities import units
from NuRadioReco.detector.RNO_G.rnog_detector import Detector

from spectralAmplitudeHistogramSimulator import spectralAmplitudeHistogramSimulator
from utilities.utility_functions import read_config



def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--sim_samples", type=int, default=5000)
    parser.add_argument("--config", default="sim/rayleigh/config.json")
    parser.add_argument("--test", default=False)
    args = parser.parse_args()
    config = read_config(args.config)

    print(config.keys())


    hist_range = np.array([0., 0.7])
    nr_bins = config["nr_bins"]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)

    detector = Detector(select_stations=args.station)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    temp = 300 * units.kelvin

    # for now use antenna bandwidth as described in the rno-g paper
    bandwidth = 0.6 - 0.15 * units.GHz

    filter_bandpass = config["filter_bandpass"]

    hist_simulator = spectralAmplitudeHistogramSimulator()
    hist_simulator.begin(args.station, detector, temp, bandwidth, filter_bandpass)

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    savedir = f"{config['save_dir']}/spec_hist/job_{date}"
    try:
        os.makedirs(savedir)
    except OSError:
        pass

    config_name = f"{savedir}/config.json"
    settings_dict = {**config, **vars(args)}
    if not os.path.isfile(config_name):
        with open(config_name, "w") as f:
            json.dump(settings_dict, f)
    savefile = f"{savedir}/spec_hist_s{args.station}_sim.pickle"
    frequencies, spec_amplitude_histograms = hist_simulator.simulate(args.sim_samples, nr_bins, hist_range, savefile=savefile)

#    frequencies, spec_amplitude_histograms = hist_simulator.simulate_single_frequency(0.4 * units.GHz, args.sim_samples, nr_bins, hist_range, savefile=savefile)


#    param, cov = curve_fit(rayleigh, bin_centers, spec_hist[0][300])
#    ray = rayleigh(bin_centers, param[0])
    plt.style.use("/user/rcamphyn/envs/gaudi.mplstyle")
#    fig, axs = plt.subplots(2, 1)
#    axs[0].stairs(spec_hist[0][300], edges = bin_edges)
#    axs[0].plot(bin_centers, rayleigh(bin_centers, param[0]))
#
#    axs[1].plot(bin_centers, (spec_hist[0][300]**2 - ray**2)/ray)

    channel_id = 0

#    f_idx = 200
#    plt.stairs(spec_amplitude_histograms[channel_id, f_idx], edges=bin_edges)
#    print(frequencies[f_idx])
#    plt.xlabel("spectral amplitude V/GHz")
#    plt.title(f"Simulated Rayleigh distribution T = {temp} K, f = {0.4} GHz, with detector")
#    plt.savefig("test")





    args.freq_lim = [0.1, 0.6]
    fig, ax = plt.subplots()
    freq_idxs = np.nonzero((args.freq_lim[0] < frequencies) & (frequencies < args.freq_lim[1]))[0]
    print(frequencies)
    print(freq_idxs)
    cmesh = ax.pcolormesh(bin_centers, frequencies[freq_idxs], spec_amplitude_histograms[channel_id, freq_idxs])
    ax.set_title(f"Simulated spectral magnitude, station {args.station}, channel {channel_id}, with detector, {args.sim_samples} samples")
    ax.set_xlabel("Spectral magnitude / V/GHz")
    ax.set_ylabel("frequencies / GHz")
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.set_label("counts")
    fig.tight_layout()
    fig.savefig("test.png")
