import argparse
from astropy.time import Time
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.detector import detector
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle

parser  =argparse.ArgumentParser()
parser.add_argument("--station", "-s", type=int, default=11)
parser.add_argument("--channel", "-c", type=int, default=0)
args = parser.parse_args()

def create_sim_event(station_id, channel_id, detector, frequencies, sampling_rate):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time(datetime.datetime(2023,1,1))
    channel = Channel(channel_id)
    channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
    station.add_channel(channel)
    event.set_station(station)
    return event, station


data_average_ft = read_pickle("/home/ruben/Documents/data/noise_study/data/average_ft/job_2025_03_11/station11/clean/average_ft_combined.pickle")
data_spectrum = data_average_ft["frequency_spectrum"]


log_level = logging.DEBUG

# det = detector.Detector(source="rnog_mongo",
#                         always_query_entire_description=False,
#                         database_connection="RNOG_public",
#                         select_stations=args.station,
#                         log_level=log_level)
# det.update(Time("2023-08-01"))

nr_samples = 2048
sampling_rate = 3.2 * units.GHz
frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

bandpass_filter = channelBandPassFilter()
# butter_filter = bandpass_filter.get_filter(frequencies, station_id=11, channel_id=0, det=det, passband=[0.1, 0.8], filter_type="butter")

scale_parameter_dir = "/home/ruben/Downloads"
generic_noise_adder = channelGenericNoiseAdder()
generic_noise_adder.begin(scale_parameter_dir=scale_parameter_dir)

nr_of_sims = 200000
frequency_spectrum_mean = np.zeros_like(frequencies, dtype=np.complex128)
event_writer = eventWriter()
event_writer.begin("test_data_driven.nur")
for _ in range(nr_of_sims):
    event, station = create_sim_event(args.station, args.channel, 0, frequencies, sampling_rate)
    generic_noise_adder.run(event, station, 0, 0, min_freq=0, max_freq=1.6, type="data-driven")
    bandpass_filter.run(event, station, 0, passband=(0.1, 0.8), filter_type="butter")
    # noise = generic_noise_adder.bandlimited_noise(0, 1.6, nr_samples, sampling_rate, amplitude=None, type="data-driven", time_domain=False,
    #                                               station_id=args.station, channel_id=args.channel)
    # noise *= butter_filter

    # frequency_spectrum_mean += np.abs(noise)
    event_writer.run(event)

    




# frequency_spectrum_mean /= nr_of_sims
# plt.style.use("gaudi")
# plt.plot(frequencies, np.abs(frequency_spectrum_mean), label = "generated noise")
# plt.plot(frequencies, np.abs(data_spectrum[0]), label = "average ft over data")
# plt.xlabel("freq / GHz")
# plt.ylabel("spectral amplitude / V/GHz")
# plt.legend()
# plt.title(f"Mean spectrum of {nr_of_sims} noise spectra \n generated by channelGenericNoiseAdder.bandlimited_noise(type='data-driven')")
# plt.show()