import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.utilities import units

from sim.thermal_noise.channelThermalNoiseAdder import channelThermalNoiseAdder



if __name__ == "__main__":

    station_id = 11
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    channel_ids = [0]

    detector = Detector(database_connection='RNOG_public', log_level=logging.NOTSET,
                        select_stations=station_id)
    detector_time = datetime.datetime(2023, 8, 1)
    detector.update(detector_time)

    specs = []

    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin(sim_library_dir="sim/library")
    for _ in range(20):
        print(_)
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())

        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)

        thermal_noise_adder.run(event, station, detector)
        specs.append(np.abs(channel.get_frequency_spectrum()))

    spec = np.mean(specs, axis=0)
    plt.plot(channel.get_frequencies(), spec)
    plt.show()
    plt.savefig("test_thermal_noise")
