import argparse
import os
import logging
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from NuRadioReco.utilities import fft
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader



def read_freq_spectrum_from_nur(files : list, event_nr=0, channel_id=0):
    event_reader = eventReader()
    event_reader.begin(files)
    for event_tmp in event_reader.run():
        event = event_tmp
        break
    station = event.get_station()
    channel = station.get_channel(channel_id)
    sampling_rate = channel.get_sampling_rate()
    nr_samples = channel.get_number_of_samples()
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    frequency_spectrum = channel.get_frequency_spectrum()
    return frequencies, frequency_spectrum







if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        default = 24)
    parser.add_argument("-r", "--run",
                        type = int, default = 1)
    parser.add_argument("-e", "--event",
                        type = int, default = 0)
    parser.add_argument("-c", "--channel",
                        type = int, default = 0)
    args = parser.parse_args()

    if args.data_dir == None:
        files = glob.glob("/home/ruben/Documents/data/noise_study/simulations/thermal_noise_traces/job_2025_02_19_16/station23/run**/events_batch*.nur", recursive=True)
        print(files)
    else:
        root_dir = glob.glob(f"{args.data_dir}/station{args.station}/run{args.run}")

#    rnog_reader = readRNOGData(log_level = logging.DEBUG)
#    rnog_reader.begin(root_dir, convert_to_voltage = True,
#                      mattak_kwargs = dict(backend = "uproot"))
#    event = rnog_reader.get_event_by_index(args.event)
#    station = event.get_station()
#    channel = station.get_channel(args.channel)
#    trace = channel.get_trace() * 1e3 #mV
#    
#
#    
#    fs = 3.2e9
#    freq = np.fft.rfftfreq(2048, d = 1/fs)
#    ft = fft.time2freq(trace, fs)

    freqs, ft = read_freq_spectrum_from_nur(files, channel_id=4)

#    rms = np.sqrt(np.mean(np.abs(ft)**2))
#    idx = np.where(np.abs(ft) > 4*rms)[0]
#    Q = 1e3
#    notch_filters = [signal.iirnotch(freq[i], Q, fs = fs) for i in idx]
#    trace_notched = signal.filtfilt(notch_filters[0][0], notch_filters[0][1], trace)
#    for notch in notch_filters[1:]:
#        trace_notched = signal.filtfilt(notch[0], notch[1], trace_notched)
#    ft_notched = fft.time2freq(trace_notched, fs)

    plt.plot(freqs, np.abs(ft), label = "frequency spectrum")
#    plt.plot(freq/1e9, np.abs(ft_notched), label = "notched")
#    plt.vlines(0.2, 0, 5e-7, color = "r", ls = "dashed", label = "200 MHz")
#    plt.hlines(4*rms, 0, 1.6, color = "gray", ls = "dashed", label = "4 $\cdot$ rms")
#    plt.scatter(freq[idx]/1e9, np.abs(ft[idx]), marker = "x", c = "r")
    plt.legend(loc = "upper right")
    plt.xlabel("freq / GHz")
    plt.ylabel("amplitude / V/GHz")
    fig_path = os.path.abspath(f"{__file__}/../../figures")
    plt.savefig(f"{fig_path}/fft")
