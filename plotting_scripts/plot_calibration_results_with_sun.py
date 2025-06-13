import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units

from fitting.spectrumFitter import spectrumFitter
from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    return frequencies, frequency_spectrum


def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file", default=None)
    parser.add_argument("--sims", "-s", help="path to sim pickle files, give in same order as noise sources, pass sum last ", nargs = "+")
    args = parser.parse_args()

    config_path = find_config(args.sims[0])
    config = read_config(config_path)
    noise_sources = config["noise_sources"]
    noise_sources.append("gal + sun")
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]
    

    spectrum_fitter = spectrumFitter(args.data, args.sims[-1])
    fit_results = spectrum_fitter.get_fit_gain()


    plt.style.use("gaudi")

    pdf = PdfPages("test_avg_ft.pdf") 
    for idx, channel_id in enumerate(channel_ids):
        fit_param = fit_results[idx]
        gain_factor = fit_param[0].value
        gain_error = fit_param[0].error

        print(channel_id)
        fig, ax = plt.subplots()
        labels = noise_sources
        if include_sum:
            labels += ["sum"]
        
        if args.data is not None:
            frequencies, frequency_spectrum = read_freq_spec_file(args.data)
            ax.plot(frequencies, frequency_spectrum[channel_id], label = "data")
        
        for i, sim in enumerate(args.sims):
            if noise_sources[i] == "electronic":
                labels[i] += f" (T = {electronic_temperature} K)"
            elif noise_sources[i] == "ice":
                labels[i] += " (T = ~240 K)"
            elif labels[i] == "sum":
                labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"
            frequencies, frequency_spectrum = read_freq_spec_file(sim)
            
            frequency_spectrum = frequency_spectrum * gain_factor
                
            ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i])
            # ax.set_ylim(0, 0.05)
            ax.legend()

        ax.set_title(f"Channel {channel_id}")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        fig.savefig(pdf, format="pdf")

    pdf.close()
