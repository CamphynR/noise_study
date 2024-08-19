import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.utilities import fft

import modules.cwFilter

def read_var_from_pickle(file_adress):
    with open(file_adress, "rb") as f:
        variables = pickle.load(f)
    return np.array(variables)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help = "path to trace pickle file")
    parser.add_argument("-s", "--station", default = 0)
    parser.add_argument("-c", "--channel", default = 0)
    args = parser.parse_args()

    fs = 3.2 * units.GHz
    traces = read_var_from_pickle(args.data)
    print(traces.shape)
    if "_mean" in args.data:
        traces = traces[args.station, args.channel]
    else:
        traces = np.mean(traces[args.station, args.channel, :, :], axis = 0)
    
    freqs = np.fft.rfftfreq(2048, d = 1/fs)
    spec_before = fft.time2freq(traces, sampling_rate = fs)
    
    filtered_trace = modules.cwFilter.filter_cws(traces)
    spec_after = fft.time2freq(filtered_trace, sampling_rate = fs)

    fig, axs = plt.subplots(1, 2, figsize = (14, 6))
    axs[0].plot(traces / units.mV)
    axs[1].plot(freqs, np.abs(spec_before))
    axs[1].plot(freqs, np.abs(spec_after))
    fig_path = os.path.abspath(f"{__file__}/../../figures")
    figname = f"{fig_path}/test_cw_filter"
    print(f"saving as {figname}")
    plt.savefig(figname, bbox_inches = "tight") 