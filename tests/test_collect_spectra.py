import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.modules.io.eventReader import eventReader

filename = "/home/rcamphyn/data/noise_study/data/spectra/job_2025_03_27_test/station23/clean/spectra_batch0.nur"
channel_id = 0

event_reader = eventReader()
event_reader.begin(filename)

i=0
for event in event_reader.run():
    i += 1
    station = event.get_station()
    channel = station.get_channel(channel_id)
    frequencies = channel.get_frequencies()
    spectrum = channel.get_frequency_spectrum()
print(f"contains {i} events")
plt.plot(frequencies, np.abs(spectrum))
plt.savefig("figures/tests/test_collect_spectra")
plt.close()
plt.plot(channel.get_times(), channel.get_trace())
plt.savefig("figures/tests/test_collect_spectra_trace")