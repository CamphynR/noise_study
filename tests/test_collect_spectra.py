import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioReco.modules.io.eventReader import eventReader

filename = "/mnt/rno-g-0/felix/ruben/data/noise_study/data/spectra/job_2025_03_28_test/station11/clean/spectra_batch0.nur"
filename_trace = "/mnt/rno-g-0/felix/ruben/data/noise_study/data/spectra/job_2025_03_28_test_trace_stored/station11/clean/spectra_batch0.nur"

event_reader = eventReader()
event_reader.begin(filename)

t0 = time.time()
i=0
for event in event_reader.run():
    i += 1
    station = event.get_station()
    for channel in station.iter_channels():
        frequencies = channel.get_frequencies()
        spectrum = channel.get_frequency_spectrum()
dt = time.time() - t0
print(f"reading took {dt} s")
print(f"contains {i} events")
print(f"time per event: {(dt/i)*1000} ms")
plt.plot(frequencies, np.abs(spectrum))
plt.savefig("figures/tests/test_collect_spectra")
plt.close()
plt.plot(channel.get_times(), channel.get_trace())
plt.savefig("figures/tests/test_collect_spectra_trace")
event_reader.end()

event_reader_trace = eventReader()
event_reader_trace.begin(filename_trace)
t0 = time.time()
i=0
for event in event_reader_trace.run():
    i += 1
    station = event.get_station()
    for channel in station.iter_channels():
        frequencies = channel.get_frequencies()
        spectrum = channel.get_frequency_spectrum()
dt = time.time() - t0
print(f"reading took {dt}")
print(f"contains {i} events")
print(f"time per event: {(dt/i)*1000} ms")