import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=[23], nargs="+")
    parser.add_argument("--channel", "-c", type=int, default=[23], nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sampling_rate = 3.2*units.GHz
    frequencies = np.fft.rfftfreq(2048, d=1./3.2*units.GHz)

    det = Detector(source="rnog_mongo",
                   select_stations=args.station)
    det_time = Time("2023-08-01")
    det.update(det_time)


    plt.style.use("retro")
    channel_id = 4
    for station_id in args.station:
        signal_chain = det.get_signal_chain_response(station_id, channel_id)
        if args.debug:
            signal_chain.plot(show=True)
        hardware_db_response = signal_chain(frequencies)
        hardware_db_response = hardware_db_response / np.max(np.abs(hardware_db_response))

        plt.plot(frequencies, np.abs(hardware_db_response), label=f"S21 measurements, station {station_id}, channel {channel_id}",
                 lw=2.)


    system_response_time = systemResonseTimeDomainIncorporator()
    system_response_time.begin(response_path="sim/library/deep_impulse_responses.json", det=0)
    response_channel_keys = ["ch2_6dB", "ch11_6dB", "ch9_6dB", "ch2"]
    channel_mapping = {2 : "ch2_6dB" ,  11 : "ch11_6dB", 9 : "ch9_6dB", 0: "ch2"}
    channel_ids = [2, 9]
    for channel_id in channel_ids:
        response = system_response_time.get_response(channel_id=channel_id)
        freqs = np.fft.rfftfreq(2048, d=1./3.2)

        plt.plot(freqs, response(freqs), label=f"Time domain measurements {channel_mapping[channel_id]}",
                 lw=2.)

    plt.minorticks_on()
    plt.grid(which="minor", alpha=0.2, ls="dashed")
    plt.legend(loc=8)
    plt.xlabel("freq / GHz")
    plt.ylabel("normalized spectral amplitude / V/GHz")
    plt.xlim(0, 1.)
#    plt.title(f"Detector response at {det_time}")
    plt.savefig("figures/tests/test_hardware.pdf", bbox_inches="tight")
    plt.close()


#    channel_id = 0
#    for station_id in args.station:
#        signal_chain = det.get_signal_chain_response(station_id, channel_id)
#        if args.debug:
#            signal_chain.plot(show=True)
#        hardware_db_sampling_rate = 3.2*units.GHz
#        hardware_db_response = signal_chain(frequencies)
#        hardware_db_response = freq2time(hardware_db_response, hardware_db_sampling_rate, n=2048)
#        hardware_db_response = -1*hardware_db_response / np.max(-1*hardware_db_response)
#        hardware_max_idx = np.argmax(hardware_db_response)
#
#        plt.plot(np.arange(-1024, 1024)/hardware_db_sampling_rate, np.roll(hardware_db_response, -1*hardware_max_idx + 1024), label=f"S21 measurements, station {station_id}, channel {channel_id}")
#
#
#    system_response_time = systemResonseTimeDomainIncorporator()
#    system_response_time.begin(response_path="sim/library/deep_impulse_responses.json", det=0)
#    channel_ids_time = [0, 4, 9]
#    sampling_rate = 3.2 * units.GHz
#    for channel_id in channel_ids_time:
#        response = system_response_time.get_response(channel_id=channel_id)
#        freqs = np.fft.rfftfreq(2048, d=1./sampling_rate)
#
#        channel_mapping = {"ch2" : [0, 1, 2, 3],
#                           "ch11_6dB" : [4, 8, 11, 21],
#                           "ch9_6dB" : [9, 10, 22, 23]}
#        channel_mapping = {key: group for group, keys in channel_mapping.items() for key in keys}
#        plt.plot(times, system_response, label=f"Time domain measurements {channel_mapping[channel_id]}")
#
#    plt.legend()
#    plt.xlabel("times / ns")
#    plt.ylabel("normalized impulse response")
#    plt.xlim(-10, 30)
#    plt.title(f"Detector response at {det_time}")
#    plt.show()
