import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


from utilities.utility_functions import read_pickle



def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--linear")
    parser.add_argument("--vc")
    args = parser.parse_args()

    plt.style.use("gaudi")

    frequency_lin, frequency_spectrum_lin, var_frequency_spectrum_lin = read_average_freq_spectrum_from_pickle(args.linear)
    frequency_vc, frequency_spectrum_vc, var_frequency_spectrum_vc = read_average_freq_spectrum_from_pickle(args.vc)
    residuals = (frequency_spectrum_lin - frequency_spectrum_vc)/frequency_spectrum_lin
    freq_selection = [0.1, 0.7]
    selection = np.logical_and(freq_selection[0] < frequency_lin, frequency_lin < freq_selection[1])
    pdf = PdfPages("figures/tests/lin_vc_comparsion.pdf")
    for channel_id in np.arange(24):
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(frequency_lin, frequency_spectrum_lin[channel_id], label = "linear cal")
        axs[0].fill_between(frequency_lin,
                         frequency_spectrum_lin[channel_id] - var_frequency_spectrum_lin[channel_id],
                         frequency_spectrum_lin[channel_id] + var_frequency_spectrum_lin[channel_id],
                         alpha=0.2,
                         label="var")
        axs[0].plot(frequency_vc, frequency_spectrum_vc[channel_id], label = "voltage cal")
        axs[0].fill_between(frequency_vc,
                         frequency_spectrum_vc[channel_id] - var_frequency_spectrum_vc[channel_id],
                         frequency_spectrum_vc[channel_id] + var_frequency_spectrum_vc[channel_id],
                         alpha=0.2,
                         label="var")
        axs[0].legend()
        axs[0].set_xlim(0.1, 0.7)
        axs[0].set_ylabel("spectral amplitude / V/GHz")
        axs[1].plot(frequency_vc[selection], residuals[channel_id][selection])
        axs[1].set_title("relative residuals")
        axs[1].set_xlabel("freq / GHz")

        fig.suptitle(f"Channel {channel_id}")
        fig.savefig(pdf, format="pdf")
        plt.close()
    pdf.close()

