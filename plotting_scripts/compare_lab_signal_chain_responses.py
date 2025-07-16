import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator
from modules.systemResponseTimeDomainIncorporator import open_response_file
from modules.systemResponseTimeDomainIncorporator import convert_response_to_spectrum




if __name__ == "__main__":
    nr_samples = 2048
    sampling_rate = 3.2*units.GHz
    frequencies = np.fft.rfftfreq(2048, d=1./3.2*units.GHz)


#    response_path = "sim/library/deep_impulse_responses.json"
#    response_path = "sim/library/v2_v3_deep_impulse_responses.json"
    response_path = "sim/library/v2_v3_surface_impulse_responses.json"
    responses = open_response_file(response_path)
#    response_channel_keys = ["ch2_6dB", "ch11_6dB", "ch9_6dB", "ch2"]
#    response_channel_keys = ["v3_ch1_62dB", "v3_ch4_62dB", "v3_ch8_62dB"]
    response_channel_keys = ["v3_ch13", "v3_ch14", "v3_ch5"]
#    response_channel_keys = list(responses.keys())
#    response_channel_keys.remove("time")

    plt.style.use("retro")
    for response_channel_key in response_channel_keys:
        if response_channel_key.endswith("times"):
            continue
        times, response = responses[f"{response_channel_key}_times"], responses[response_channel_key]
        freqs = np.fft.rfftfreq(2048, d=1./3.2)
        system_response = convert_response_to_spectrum(responses, response_channel_key)
#        sampling_rate = 1./np.diff(times[0:2])
#    
#        time_range = [-5, 30]
#        time_selection = np.logical_not(np.logical_and(time_range[0] < times, times < time_range[1]))
#
#        response = response(times)
#        response[time_selection] = 0
#
#        frequencies_system_response = np.fft.rfftfreq(len(times), d=1./sampling_rate)
#        system_response = time2freq(response, np.diff(times[0:2]))
#        system_response = np.abs(system_response)
#        system_response = system_response/np.max(system_response[1:])

#        plt.plot(frequencies_system_response[1:], system_response[1:], label=f"Time domain measurements {response_channel_key}", lw=2.)
        plt.plot(freqs, system_response(freqs), label=f"{response_channel_key}", lw=2.)

    plt.legend(bbox_to_anchor=(1., 1.))
    plt.tight_layout()
    plt.xlabel("freq / GHz")
    plt.ylabel("normalized spectral amplitude / V/GHz")
    plt.xlim(0, 1.)
    plt.title(f"Lab measured signal chain")
    plt.savefig("test_fr")
    plt.close()



    for response_channel_key in response_channel_keys:
        if response_channel_key.endswith("times"):
            continue
        times, response = responses[f"{response_channel_key}_times"], responses[response_channel_key]
        sampling_rate = 1./np.diff(times[0:2])
    
        time_range = [-100, 100]
        time_selection = np.logical_and(time_range[0] < times, times < time_range[1])
        times = times[time_selection]
        system_response = response(times)/np.max(response(times))

        plt.plot(times, system_response, label=f"{response_channel_key}", lw=2.)

    plt.legend(bbox_to_anchor=(1., 1.))
    plt.xlabel("times / ns")
    plt.ylabel("normalized impulse response")
    plt.xlim(-10, 30)
    plt.title(f"Lab measured signal chain")
    plt.savefig("test_imp")
