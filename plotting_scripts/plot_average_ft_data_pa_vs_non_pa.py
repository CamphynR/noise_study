from astropy.time import Time
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units


from utilities.utility_functions import read_pickle



def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"], contents["header"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    args = parser.parse_args()

    channel_types = {"pa" : [0, 1, 2, 3], "helper" : [5, 6, 7, 9, 10, 22, 23]}
    

    plt.style.use("retro")
    color_list = plt.rcParams["axes.prop_cycle"].by_key()['color']
    colors = {"pa" : color_list[0], "helper" : color_list[1]}
    fig, ax = plt.subplots()
    frequency, frequency_spectrum, var_frequency_spectrum, header = read_average_freq_spectrum_from_pickle(args.pickle)
    freq = 600*units.MHz
    freq_idx = np.where(np.isclose(freq, frequency))[0]
#    frequency_spectrum = frequency_spectrum / frequency_spectrum[:, freq_idx]
    std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
    for channel_type in channel_types:
        for channel_id in channel_types[channel_type]:
            ax.plot(frequency, frequency_spectrum[channel_id], label = f"channel {channel_id}", lw=1., color=colors[channel_type])
#            ax.fill_between(frequency,
#                             frequency_spectrum[channel_id] - std_frequency_spectrum[channel_id],
#                             frequency_spectrum[channel_id] + std_frequency_spectrum[channel_id],
#                             alpha=0.2,
#                            color=colors[channel_type])

        ax.grid(which="major", alpha=0.8)
        ax.minorticks_on()
        ax.grid(which="minor", alpha=0.2, ls="dashed")
        ax.legend()
        ax.set_xlim(0., 1.)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_title(f"Channel {channel_id}")
    fig.tight_layout()
    fig.savefig("test_data.pdf", bbox_inches="tight")
