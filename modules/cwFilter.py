import numpy as np
from scipy import signal
from NuRadioReco.utilities import units
from NuRadioReco.utilities import fft

"""
Contains function to filter continuous wave out of the signal
functions should work on a per event basis to comply with the iteration methods used in readRNOGData
"""


def find_frequency_peaks(trace : np.ndarray, threshold = 4):
    """
    Function fo find the frequency peaks in the real fourier transform of the input trace,

    Parameters
    ----------
    trace : np.ndarray
        waveform (shape: [24,2048])
    threshold: int, default = 4
        threshold for peak definition. A peak is defined as a point in the frequency spectrum
        that exceeds threshold * rms(real fourier transform)
    
    Returns
    -------
    freq: np.ndarray
        frequencies at which a peak was found
    """
    fs = 3.2e9 * units.Hz
    freq = np.fft.rfftfreq(2048, d = 1/fs)
    ft = fft.time2freq(trace, fs)
    
    rms = np.sqrt(np.mean(np.abs(ft)**2))
    peak_idxs = np.where(np.abs(ft) > threshold * rms)[0]

    return freq[peak_idxs]


def filter_cws(trace, Q = 1e3, threshold = 4, fs = 3.2e9 * units.Hz):
    """
    Function that applies a notch filter at the frequency peaks of a given time trace
    using the scipy library

    Parameters
    ----------
    trace : np.ndarray
        waveform (shape: [24,2048])
    Q: int, default = 1000
        quality factor of the notch filter, defined as the ratio f0/bw, where f0 is the centre frequency
        and bw the bandwidth of the filter at (f0,-3 dB)
    threshold: int, default = 4
        threshold for peak definition. A peak is defined as a point in the frequency spectrum
        that exceeds threshold * rms(real fourier transform)
    fs: float, default = 3.2e9 Hz
        sampling frequency of the RNO-G DAQ
    """
    freqs = find_frequency_peaks(trace, threshold=threshold)

    if len(freqs) !=0:
        notch_filters = [signal.iirnotch(freq, Q, fs = fs) for freq in freqs]
        notch_filters = np.array(notch_filters).reshape(-1, 6)
        trace_notched = signal.sosfiltfilt(notch_filters, trace)
        return trace_notched
    return trace


def plot_trace(channel, ax, fs = 3.2e9 * units.Hz, label = None, plot_kwargs = dict()):
    """
    Function to plot trace of given channel
    
    Parameters
    ----------
    channel : NuRadio channel class
        channel from which to get trace
    ax : matplotlib.axes
        ax on which to plot
    fs : float, default = 3.2e9 Hz
        sampling frequency of the RNO-G DAQ
    label : string
        plotlabel
    plot_kwargs : dict
        options for plotting
    """
    times = np.arange(2048)/fs / units.ns
    trace = channel.get_trace()

    legendloc = 2

    ax.plot(times, trace, label = label, **plot_kwargs)
    ax.set_xlabel("time / ns")
    ax.set_ylabel("trace / V")
    ax.legend(loc = legendloc)


def plot_ft(channel, ax, label = None, plot_kwargs = dict()):
    """
    Function to plot real frequency spectrum of given channel
    
    Parameters
    ----------
    channel : NuRadio channel class
        channel from which to get trace
    ax : matplotlib.axes
        ax on which to plot
    label : string
        plotlabel
    plot_kwargs : dict
        options for plotting
    """
    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()

    legendloc = 2

    ax.plot(freqs, np.abs(spec), label = label, **plot_kwargs)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    ax.legend(loc = legendloc)


class cwFilter():
    """
    cwFilter class to apply the module as defined by NuRadio module syntax
    """
    def __init__(self):
        pass

    def begin(self, Q = 1e3, threshold = 4, fs = 3.2e9 * units.Hz):
        self.Q = Q
        self.threshold = threshold
        self._fs = fs

    def run(self, event, station, det):
        for channel in station.iter_channels():
            trace = channel.get_trace()
            trace_fil = filter_cws(trace, Q = self.Q, threshold = self.threshold, fs = self._fs)
            channel.set_trace(trace_fil, self._fs)
        
# Standard test for people playing around with module settings, applies the module as one would in a data reading pipelin
# using one event in RNO_G_DATA (choose station and run) as a test
if __name__ == "__main__":
    import os
    import logging
    import argparse
    import matplotlib.pyplot as plt
   
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

    parser = argparse.ArgumentParser(prog = "%(prog)s", usage = "cw filter test")
    parser.add_argument("--station", type = int, default = 24)
    parser.add_argument("--run", type = int, default = 1)

    parser.add_argument("--Q", type = int, default = 1e3)
    parser.add_argument("--threshold", type = int, default = 4)
    parser.add_argument("--fs", type = float, default = 3.2e9 * units.Hz)
    
    parser.add_argument("--save_dir", type = str, default = None,
                        help = "Directory where to save plot produced by the test.\
                                If None, saves to NuRadioReco test directory")

    args = parser.parse_args()
    
    data_dir = os.environ["RNO_G_DATA"]
    rnog_reader = readRNOGData(log_level = logging.DEBUG)

    root_dirs = f"{data_dir}/station{args.station}/run{args.run}"
    rnog_reader.begin(root_dirs,
                      convert_to_voltage = True,                    # linear voltage calibration
                      mattak_kwargs = dict(backend = "uproot"))

    cwFilter = cwFilter()
    cwFilter.begin(Q = args.Q, threshold = args.threshold, fs = args.fs)

    for event in rnog_reader.run():
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)

        fig, axs = plt.subplots(1, 2, figsize = (14, 6))
        plot_trace(station.get_channel(0), axs[0], label = "before")
        plot_ft(station.get_channel(0), axs[1], label = "before")
        cwFilter.run(event, station, det = 0)
        plot_trace(station.get_channel(0), axs[0], label = "after")
        plot_ft(station.get_channel(0), axs[1], label = "after")
        
        if args.save_dir is None:
            fig_dir = os.path.abspath("{__file__}/../test")
        else:
            fig_dir = args.save_dir

        fig.savefig(f"{fig_dir}/test_cw_filter", bbox_inches = "tight")
        break