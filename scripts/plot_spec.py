import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units

from constants import sample_rate, nr_of_samples

def read_var_from_pickle(file_adress):
    with open(file_adress, "rb") as f:
        variables = pickle.load(f)
    return np.array(variables)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help = "path to spec pickle file")
    parser.add_argument("-b", "--before", help = "path to spec pickle file to which the original spec can be compared",
                        default = "/user/rcamphyn/noise_study/variable_lists/spec_lists/spec_s24_unclean_mean.pickle")
    parser.add_argument("-s", "--station", type = int, default = 0)
    parser.add_argument("-c", "--channel", type = int, default = 0)

    args = parser.parse_args()

    spec = read_var_from_pickle(args.data)[args.station, args.channel]
    print(f"spec contains shape: {spec.shape}")

    spec_before = read_var_from_pickle(args.before)[args.station, args.channel]
    print(f"spec_before contains shape: {spec_before.shape}")
    
    freqs = np.fft.rfftfreq(nr_of_samples, d = (1./sample_rate))

    fig, ax = plt.subplots(1, figsize = (12, 8))

    ax.plot(freqs/units.GHz, spec_before, label = "before")
    ax.plot([0.403, 0.403], [0, np.max(spec_before)], color = "gray", alpha = 0.5, ls = "dashed", label = "wheather balloon")
    
    ax2 = ax.twinx()
    ax2.plot(freqs/units.GHz, spec, color = "orange", label = "after")
    ax2.tick_params(axis = "y", labelcolor = "orange")
    ax2.set_ylabel("amplitude / V/GHz", color = "orange")
    ax2.legend(loc = "best")

    ax.set_xlim(0.1, 0.7)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    ax.legend(loc = 2)

    fig.suptitle("Mean spectrum of station 24, handcarry 22 (contains 166885 events)")
    fig.tight_layout()
    figdir = os.path.abspath(f"{__name__}/../figures")
    figname = f"{figdir}/spec"
    print(f"saving as {figname}") 
    fig.savefig(figname, bbox_inches = "tight")