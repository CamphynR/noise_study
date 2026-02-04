import argparse
import matplotlib.pyplot as plt
import natsort
import numpy as np


from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()

    args.pickles = natsort.natsorted(args.pickles)
    channel_id = 0

    times = []
    vrms = []
    for pickle in args.pickles:
        content = read_pickle(pickle)
        if np.any(content["vrms"] < 0.002):
            continue
        times.append(content["header"]["begin_time"][0])
        vrms.append(content["vrms"])

    vrms = np.array(vrms)
    times_epoch = [t.unix for t in times]
    times_string = [t.strftime("%B %d")  for t in times]

    plt.style.use("retro")

    fig, ax = plt.subplots()
    ax.plot(times_epoch, vrms[:, channel_id] / units.mV)
    ax.set_xlabel("time")
    ax.set_ylabel("vrms / mV")
    nr_ticks = 10
    step = int(len(times) / nr_ticks)
    ax.set_xticks(times_epoch[::step], times_string[::step], rotation=45)
    fig.tight_layout()
    fig.savefig("figures/vrms.png")
