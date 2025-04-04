import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator

try:
    plt.style.use("gaudi")
except:
    pass








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=[23], nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sampling_rate = 3.2*units.GHz
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

        plt.plot(frequencies, np.abs(hardware_db_response), label=f"S21 measurements, station {station_id}, channel {channel_id}")


    system_response_time = systemResonseTimeDomainIncorporator()
    system_response_time.begin(response_path="sim/library/deep_impulse_responses.json")
    response_channel_keys = ["ch2_6dB", "ch11_6dB", "ch9_6dB", "ch2"]
    for response_channel_key in response_channel_keys:
        times, response = system_response_time.get_response(response_channel_key=response_channel_key)
        sampling_rate = 1./np.diff(times[0:2])
    
        time_range = [-5, 30]
        time_selection = np.logical_not(np.logical_and(time_range[0] < times, times < time_range[1]))

        response = response(times)
        response[time_selection] = 0

        frequencies_system_response = np.fft.rfftfreq(len(times), d=1./sampling_rate)
        system_response = time2freq(response, np.diff(times[0:2]))
        system_response = np.abs(system_response)
        system_response = system_response/np.max(system_response[1:])

        plt.plot(frequencies_system_response[1:], system_response[1:], label=f"Time domain measurements {response_channel_key}")

    plt.legend(loc=8)
    plt.xlabel("freq / GHz")
    plt.ylabel("normalized spectral amplitude / V/GHz")
    plt.xlim(0, 1.)
    plt.title(f"Detector response at {det_time}")
    plt.show()
    plt.close()


    channel_id = 0
    for station_id in args.station:
        signal_chain = det.get_signal_chain_response(station_id, channel_id)
        if args.debug:
            signal_chain.plot(show=True)
        hardware_db_sampling_rate = 3.2*units.GHz
        hardware_db_response = signal_chain(frequencies)
        hardware_db_response = freq2time(hardware_db_response, hardware_db_sampling_rate, n=2048)
        hardware_db_response = -1*hardware_db_response / np.max(-1*hardware_db_response)
        hardware_max_idx = np.argmax(hardware_db_response)

        plt.plot(np.arange(-1024, 1024)/hardware_db_sampling_rate, np.roll(hardware_db_response, -1*hardware_max_idx + 1024), label=f"S21 measurements, station {station_id}, channel {channel_id}")


    system_response_time = systemResonseTimeDomainIncorporator()
    system_response_time.begin(response_path="sim/library/deep_impulse_responses.json")
    channel_ids_time = [0, 4, 9]
    for channel_id in channel_ids_time:
        times, response = system_response_time.get_response(channel_id=channel_id)
        sampling_rate = 1./np.diff(times[0:2])
    
        time_range = [-100, 100]
        time_selection = np.logical_and(time_range[0] < times, times < time_range[1])
        times = times[time_selection]
        system_response = response(times)/np.max(response(times))

        channel_mapping = {"ch2" : [0, 1, 2, 3],
                           "ch11_6dB" : [4, 8, 11, 21],
                           "ch9_6dB" : [9, 10, 22, 23]}
        channel_mapping = {key: group for group, keys in channel_mapping.items() for key in keys}
        plt.plot(times, system_response, label=f"Time domain measurements {channel_mapping[channel_id]}")

    plt.legend()
    plt.xlabel("times / ns")
    plt.ylabel("normalized impulse response")
    plt.xlim(-10, 30)
    plt.title(f"Detector response at {det_time}")
    plt.show()
