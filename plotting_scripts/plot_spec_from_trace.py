import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units

def read_var_from_pickle(file_adress):
    with open(file_adress, "rb") as f:
        variables = pickle.load(f)
    return np.array(variables)

def plot_spec_from_trace(trace, fs = 3.2 * units.GHz, nr_of_samples = 2048):
    freqs = np.fft.rfftfreq(nr_of_samples, d = (1./fs))
    spec = fft.time2freq(trace, sampling_rate=fs)
    return freqs, spec

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help = "path to pickle file")
    parser.add_argument("-s", "--station", default = 0, type = int)
    parser.add_argument("-c", "--channel", default = 0, type = int)
    args = parser.parse_args()

    traces = read_var_from_pickle(args.data)
    print(f"trace contains shape: {traces.shape}")

    freqs, spec = plot_spec_from_trace(traces)
    print(f"spec contains shape: {spec.shape}")

    if "_mean" not in args.data:
        spec_abs_mean = np.mean(np.abs(spec[args.station, args.channel, :, :]), axis = 0)
    else:
        print("Be careful when using mean traces since noise will interfere destructively, yielding nonsensical results")
        spec_abs_mean = np.abs(spec[args.station, args.channel])

    fig, ax = plt.subplots(figsize = (12, 8))

    ax.plot(freqs / units.MHz, spec_abs_mean)
    ax.plot([403, 403], [0, np.max(spec_abs_mean)], color = "gray", ls = "dashed", label = "wheather balloon")
    ax.set_xlim(100, 700)
    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("amplitude / V/GHz")
    figdir = os.path.abspath(f"{__name__}/../figures")
    filename = f"{figdir}/spec_from_trace"
    if "unclean" in args.data:
        filename += "_unclean"
    else:
        filename += "_clean"
    print(f"saving as {filename}")
    fig.savefig(filename, bbox_inches = "tight")