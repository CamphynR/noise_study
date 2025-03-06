import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator

try:
    plt.style.use("gaudi")
except:
    pass








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=23, nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    frequencies = np.fft.rfftfreq(2048, d=1./3.2*units.GHz)

    det = Detector(source="rnog_mongo",
                   select_stations=args.station)
    det_time = Time("2023-08-01")
    det.update(det_time)


    channel_id = 0
    for station_id in args.station:
        signal_chain = det.get_signal_chain_response(station_id, channel_id)
        if args.debug:
            signal_chain.plot(show=True)
        hardware_db_response = signal_chain(frequencies)
        hardware_db_response = hardware_db_response / np.max(np.abs(hardware_db_response))

        plt.plot(frequencies, np.abs(hardware_db_response), label=f"S21 measurements, station {station_id}")


    system_response_time = systemResonseTimeDomainIncorporator()
    system_response_time.begin(response_path="sim/library/deep_impulse_responses.json")
    times, response = system_response_time.get_response(channel_id=channel_id)
    sampling_rate = 1./np.diff(times[0:2])
    frequencies_system_response = np.fft.rfftfreq(len(response(times)), d=1./sampling_rate)
    system_response = time2freq(response(times), np.diff(times[0:2]))
    system_response = system_response/np.max(np.abs(system_response))

    plt.plot(frequencies_system_response, np.abs(system_response), label="Time domain measurements")
    plt.xlim(frequencies[0], frequencies[-1])
    plt.legend()
    plt.xlabel("freq / GHz")
    plt.ylabel("spectral amplitude / V/GHz")
    plt.title(f"Channel {channel_id}, detector response at {det_time}")
    plt.show()