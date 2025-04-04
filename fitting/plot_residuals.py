import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    return frequencies, frequency_spectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file")
    parser.add_argument("--sum_sim", "-sum", help="path to ice pickle files")
    args = parser.parse_args()

    config_path = find_config(args.sum_sim)
    config = read_config(config_path)
    channels_to_include = config["channels_to_include"]

    frequencies, data_spectrum = read_freq_spec_file(args.data)
    sum_frequencies, sum_spectrum = read_freq_spec_file(args.sum_sim)

    assert np.all(sum_frequencies == frequencies)


    residuals = np.subtract(data_spectrum, sum_spectrum)

    plt.style.use("gaudi")
    
    pdf = PdfPages("figures/data_sim_residuals.pdf")

    for channel_id in channels_to_include:
        fig, ax = plt.subplots()
        ax.plot(frequencies, residuals[channel_id])
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spec amplitude / V/GHz")
        ax.set_title(f"channel {channel_id}")
        fig.savefig(pdf, format="pdf")

    pdf.close()
