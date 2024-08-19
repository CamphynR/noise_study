import numpy as np
from scipy import signal
import mattak.Dataset
from NuRadioReco.utilities import units
from NuRadioReco.utilities import fft

"""
Contains function to filter continuous wave out of the signal
functions should work on a per event basis to comply with the iteration methods used in readRNOGData
"""

def find_frequency_peaks(trace : np.ndarray, threshold = 4):
    """
    Parameters
    ----------
    wf : np.ndarray
        waveform (shape: [24,2048])
    """
    fs = 3.2e9 * units.Hz
    freq = np.fft.rfftfreq(2048, d = 1/fs)
    ft = fft.time2freq(trace, fs)
    
    rms = np.sqrt(np.mean(np.abs(ft)**2))
    peak_idxs = np.where(np.abs(ft) > threshold * rms)[0]

    return freq[peak_idxs]

def filter_cws(trace, Q = 1e3, fs = 3.2e9 * units.Hz):
    freqs = find_frequency_peaks(trace)

    if len(freqs) !=0:
        notch_filters = [signal.iirnotch(freq, Q, fs = fs) for freq in freqs]
        trace_notched = signal.filtfilt(notch_filters[0][0], notch_filters[0][1], trace)
        for notch in notch_filters[1:]:
            trace_notched = signal.filtfilt(notch[0], notch[1], trace_notched)
        return trace_notched

    return trace


class cwFilter():
    def __init__(self):
        pass

    def begin(self, fs = 3.2e9 * units.Hz):
        self._fs = fs

    def run(self, event, station, det):
        for channel in station.iter_channels():
            trace = channel.get_trace()
            trace_fil = filter_cws(trace, fs = self._fs)
            channel.set_trace(trace_fil, self._fs)
        
# test
if __name__ == "__main__":
    import os
    import logging
    import argparse
    import matplotlib.pyplot as plt
   
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    from plotting_functions import plot_ft

    parser = argparse.ArgumentParser(prog = "%(prog)s", usage = "cw filter test")
    parser.add_argument("--station", type = int, default = 24)
    parser.add_argument("--run", type = int, default = 1)
    args = parser.parse_args()
    
    data_dir = os.environ["RNO_G_DATA"]
    rnog_reader = readRNOGData(log_level = logging.DEBUG)

    root_dirs = f"{data_dir}/station{args.station}/run{args.run}"
    rnog_reader.begin(root_dirs,
                      convert_to_voltage = True,
                      mattak_kwargs = dict(backend = "uproot"))

    cwFilter = cwFilter()
    cwFilter.begin(fs = 3.2e9)

    for event in rnog_reader.run():
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)

        fig, ax = plt.subplots()
        plot_ft(station.get_channel(0), ax, label = "before")
        cwFilter.run(event, station)
        plot_ft(station.get_channel(0), ax, label = "after")
        fig_dir = os.path.abspath("{__file__}/../figures")
        fig.savefig(f"{fig_dir}/test_cw_filter")
        break