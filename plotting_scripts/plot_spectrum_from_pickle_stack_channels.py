from astropy.time import Time
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


from utilities.utility_functions import read_pickle



def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"], contents["header"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    parser.add_argument("--channels", "-c", nargs="+", default=[0], type=int)
    args = parser.parse_args()

    channel_mapping = {
            "PA" : [0, 1, 2, 3],
            "PS Vpol" : [5, 6, 7],
            "PS Hpol" : [4, 8],
            "helper Vpol" : [9, 10, 22, 23],
            "helper HPol" : [11, 21],
            "LPDA up" : [13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20]
            }

    colors = ["blue", "orange", "green", "black", "red", "yellow", "purple"]
    colors = {key : value for key, value in zip(channel_mapping, colors)}
    channel_mapping = {key : value for (value, chlist) in zip(channel_mapping.keys(), channel_mapping.values()) for key in chlist}

    plt.style.use("retro")
    pdf = PdfPages("figures/tests/test_average_ft_batch.pdf")
    fig, ax = plt.subplots()
    for channel_id in args.channels:
        channel_type = channel_mapping[channel_id]
        frequency, frequency_spectrum, var_frequency_spectrum, header = read_average_freq_spectrum_from_pickle(args.pickle)
        std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
        ax.plot(frequency, frequency_spectrum[channel_id], label = channel_type, color=colors[channel_type])
        ax.fill_between(frequency,
                         frequency_spectrum[channel_id] - std_frequency_spectrum[channel_id],
                         frequency_spectrum[channel_id] + std_frequency_spectrum[channel_id],
                         alpha=0.5,
                        color=colors[channel_type])

        ax.legend(ncols=2)
        ax.set_xlim(0., 1.)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_title(f"Average frequency spectra station 11 season 2023")
    fig.savefig("figures/tests/test_average_ft_channels")
