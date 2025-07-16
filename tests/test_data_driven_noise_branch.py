import argparse
from astropy.time import Time
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.detector import detector
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle

parser = argparse.ArgumentParser()
parser.add_argument("--station", "-s", type=int, default=11)
parser.add_argument("--channel", "-c", type=int, default=0)
args = parser.parse_args()


data_average_ft = read_pickle("/home/ruben/Documents/data/noise_study/data/average_ft/job_2025_03_11/station11/clean/average_ft_combined.pickle")
data_spectrum = data_average_ft["frequency_spectrum"]

data_spec_hist = read_pickle("/home/ruben/Documents/data/noise_study/data/spec_hist/job_2025_03_06_test/station11/clean/spec_amplitude_histograms_combined.pickle")
bin_centres = data_spec_hist["header"]["bin_centres"]
bin_width = np.diff(bin_centres)[0]
bin_edges = np.zeros(len(bin_centres) + 1)
bin_edges[:-1] = bin_centres - bin_width
bin_edges[-1] = bin_centres[-1] + bin_width
histograms = data_spec_hist["spec_amplitude_histograms"]
#normalize
normalization_factor = np.sum(histograms, axis = -1) * np.diff(bin_edges)[0]
histograms = np.divide(histograms, normalization_factor[:, :, np.newaxis])



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

scale_parameter_dir = "/home/ruben/Documents/data/noise_study/data/spec_hist/job_2025_03_06_test/station11/clean"
generic_noise_adder = channelGenericNoiseAdder()
generic_noise_adder.begin(scale_parameter_dir=scale_parameter_dir)

nr_of_sims = 10000

# test_dir = f"testing/station{args.station}/run0"
# os.makedirs(test_dir, exist_ok=True)
# test_path = f"{test_dir}/events_batch0.nur"
# events = []
# if os.path.exists(test_path):
#     _, station = create_sim_event(args.station, args.channel, 0, frequencies, sampling_rate)
#     event_reader = eventReader()
#     event_reader.begin(test_path)
#     for event in event_reader.run():
#         events.append(event)
# else:
#     event_writer = eventWriter()
#     event_writer.begin(test_path, max_file_size=2048)
#     for _ in range(nr_of_sims):
#         event, station = create_sim_event(args.station, args.channel, 0, frequencies, sampling_rate)
#         generic_noise_adder.run(event, station, 0, 0, min_freq=0, max_freq=1., type="data-driven")
#         events.append(event)
#         event_writer.run(event)

    

frequency_spectrum_mean = np.zeros_like(frequencies, dtype=np.complex128)
frequency_spectrum_var = np.zeros_like(frequencies, dtype=np.complex128)
freq_idx = np.where(np.isclose(frequencies, 312.5 * units.MHz))
noises = []
for _ in range(nr_of_sims):
    noise = np.abs(generic_noise_adder.bandlimited_noise(0., 1., nr_samples, sampling_rate, 0, type="data-driven", time_domain=False, station_id=args.station, channel_id=args.channel))
    print(noise.shape)
    frequency_spectrum_mean += noise
    frequency_spectrum_var += noise**2
    noises.append(np.squeeze(noise))
test_i = 200
noises = np.array(noises)
plt.hist(noises[:, test_i], bins=bin_edges, rwidth=1., density=True, label="sim")
plt.stairs(histograms[0][test_i], edges=bin_edges, label = "data")
plt.legend()
plt.xlabel("spectral amplitude / V/GHz")
plt.ylabel("Counts")
plt.yscale("log")
plt.title(f"Frequency = {frequencies[test_i]}")
plt.show()
plt.close()
# filt = bandpass_filter.get_filter(frequencies, args.station, args.channel, 0, passband=(0.1, 0.8), filter_type="butter", order=2)
# spectra_list = []
# for event in events:
#     spec = np.abs(event.get_station(args.station).get_channel(args.channel).get_frequency_spectrum())
#     # spec = spec * np.abs(filt)
#     spectra_list.append(spec)
#     frequency_spectrum_mean += spec
#     frequency_spectrum_var += spec**2

# spectra_list = np.array(spectra_list)
# print(spectra_list.shape)
# print(np.argmax(spectra_list))
# max_idx = np.argmax(spectra_list)
# max_idx = max_idx // 1025
# plt.plot(frequencies, spectra_list[max_idx])
# plt.xlabel("frequencies / GHz")
# plt.ylabel("spectral amplitude / V/GHz")
# plt.show()
# plt.close()

frequency_spectrum_mean /= nr_of_sims
frequency_spectrum_var /= nr_of_sims
frequency_spectrum_var = frequency_spectrum_var - frequency_spectrum_mean**2

freq_range = [0.2, 0.7]
selection = np.logical_and(freq_range[0] < frequencies, frequencies < freq_range[1])

plt.style.use("gaudi")
plt.plot(frequencies[selection], np.abs(frequency_spectrum_mean)[selection], label = "generated noise")
plt.fill_between(frequencies[selection], frequency_spectrum_mean[selection] - frequency_spectrum_var[selection], frequency_spectrum_mean[selection] + frequency_spectrum_var[selection], alpha=0.5)
plt.plot(frequencies[selection], np.abs(data_spectrum[0])[selection], label = "average ft over data")
plt.xlabel("freq / GHz")
plt.ylabel("spectral amplitude / V/GHz")
# plt.yscale("log")
plt.legend()
plt.title(f"Mean spectrum of {nr_of_sims} noise spectra \n generated by channelGenericNoiseAdder.bandlimited_noise(type='data-driven')")
plt.savefig("tests/data_driven_noise/test.png")
plt.show()