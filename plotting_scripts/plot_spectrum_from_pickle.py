from astropy.time import Time
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


from NuRadioReco.utilities import fft, units
from utilities.utility_functions import read_pickle



def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"], contents["header"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle", nargs="+")
    parser.add_argument("--labels", default=None, nargs="+")
    args = parser.parse_args()
    
    if args.labels is None:
        labels = list(np.arange(len(args.pickle)))
    else:
        labels = args.labels

    plt.style.use("retro")
    pdf = PdfPages("figures/tests/test_average_ft_batch.pdf")
    for channel_id in np.arange(24):
        if len(args.pickle) == 2:
            fig, (ax, ax_res) = plt.subplots(2, 1, sharex=True, height_ratios=(2, 1))
        else:
            fig, ax = plt.subplots()
        for i, pickle in enumerate(args.pickle):
            frequency, frequency_spectrum, var_frequency_spectrum, header = read_average_freq_spectrum_from_pickle(pickle)
            nr_channels = len(frequency_spectrum)
            std_on_mean_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
#            std_frequency_spectrum = np.sqrt(var_frequency_spectrum)
            if args.labels is None:
                try:
                    labels[i] = header["begin_time"][0].datetime.strftime("%B") #+ " - " + header["end_time"][0].datetime.strftime("%B")
#                    labels[i] = pickle.split("run")[-1].split(".pickle")[0]
                except:
                    labels[i] = Time(header["begin_time"], format="unix").strftime("%d %B")
            ax.plot(frequency, frequency_spectrum[channel_id], label = labels[i], lw=2.)
            ax.fill_between(frequency,
                             frequency_spectrum[channel_id] - std_on_mean_frequency_spectrum[channel_id],
                             frequency_spectrum[channel_id] + std_on_mean_frequency_spectrum[channel_id],
                             alpha=0.5)
            sampling_rate = 3.2 * units.GHz
            trace_tmp = fft.freq2time(frequency_spectrum[channel_id], sampling_rate)
        if len(args.pickle) == 2:
            frequency_prev, frequency_spectrum_prev, var_frequency_spectrum_prev, header_prev = read_average_freq_spectrum_from_pickle(args.pickle[0])
            frequency_spectrum[frequency_spectrum==0] = 1e-9
            residuals = (frequency_spectrum - frequency_spectrum_prev) / frequency_spectrum
            freq_range = [0.01, 0.3]
            selection = np.logical_and(freq_range[0] < frequency, frequency < freq_range[1])
            ax_res.plot(frequency[selection], residuals[channel_id][selection])
            ax_res.set_ylim(-1., 1.)
            ax_res.set_xlabel("freq / GHz")
            ax_res.set_title("Relative residuals")


        ax.legend(loc="upper right")
        ax.set_xlim(0., 1.)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_title(f"Channel {channel_id}")
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()
        if channel_id == nr_channels - 1:
            break
    pdf.close()
