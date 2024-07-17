import numpy as np

def plot_ft(channel, ax, fs = 3.2e9, label = None, secondary = False, plot_kwargs = dict()):
    from NuRadioReco.utilities import fft

    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()

    legendloc = 2
    if secondary:
        ax = ax.twinx()
        legendloc = 1

    ax.plot(freqs, np.abs(spec), label = label, **plot_kwargs)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    ax.legend(loc = legendloc)