import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare

from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle, read_config

def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json")
args = parser.parse_args()
config = read_config(args.config)


hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)
bin_centres = bin_edges[:-1] + np.diff(bin_edges[0:2])/2

data_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spec_hist/job_2025_01_12_no_cw/station23/clean/spec_amplitude_histograms.pickle"

data = read_pickle(data_dir)
print(data.keys())

frequencies = data["freq"]
spec_amplitude_histograms = data["spec_amplitude_histograms"]

channel_id = 0
freq = 400 * units.MHz
freq_idx = np.where(frequencies == freq)
print(spec_amplitude_histograms.shape)

# convert hists to pdf
normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis],out=np.zeros_like(spec_amplitude_histograms), where=normalization_factor[:, :, np.newaxis]!=0)

spec_amplitude_histograms = np.squeeze(spec_amplitude_histograms[channel_id, freq_idx])
print(spec_amplitude_histograms.shape)

sigma_guess = 0.01
param, cov = curve_fit(rayleigh, bin_centres, spec_amplitude_histograms, p0 = sigma_guess)

fig, ax = plt.subplots()
ax.stairs(spec_amplitude_histograms, edges=bin_edges, label = "data")
ax.plot(bin_centres, rayleigh(bin_centres, param), label = "Rayleigh pdf")
ax.legend()
ax.set_xlabel("Spectral amplitude / V/GHz")
ax.set_ylabel("N")
ax.text(0.9, 0.9, f"parameter value = {param[0]:.2E} +- {cov[0,0]:.2E}", va="top", ha="right", transform=ax.transAxes)
fig.suptitle("Spectral amplitude distribution, station 23, frequency 400 MHz")

fig.tight_layout()
fig.savefig("quick_test", dpi = 400)
