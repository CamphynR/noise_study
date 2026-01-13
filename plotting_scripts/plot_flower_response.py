import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import units

from modules.systemResponseTimeDomainIncorporator import open_response_file, convert_response_to_spectrum, rescale_response



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    response_path = "/user/rcamphyn/noise_study/sim/library/flower_impulse_responses.json"
    deep_response_path = "/user/rcamphyn/noise_study/sim/library/deep_impulse_responses.json"

    key = "ch2"
    impulse_response = open_response_file(response_path)
    gain, phase = convert_response_to_spectrum(impulse_response, key)

    deep_key = "ch2"
    deep_impulse_response = open_response_file(deep_response_path)
    deep_gain, deep_phase = convert_response_to_spectrum(deep_impulse_response, deep_key)
    deep_gain, deep_max = rescale_response(deep_gain, return_norm=True)
    
    nr_samples = 512
    sampling_rate= 0.472 * units.GHz
    frequencies_flower = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    nr_samples = 2048
    sampling_rate= 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    HARDCODED_CALIBRATION=565.5877570891539


    adc_to_v_radiant = 2.5 / 4098
    adc_to_v_flower = 2. / 256

    
    plt.style.use("retro")
    fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
    axs[0].plot(frequencies, adc_to_v_flower * gain(frequencies)/deep_max, label=f"flower gain ({key})")
    axs[0].plot(frequencies, adc_to_v_radiant * deep_gain(frequencies), label=f"radiant gain ({deep_key})")
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Gain of response")
    axs[0].legend()
    axs[1].plot(frequencies, phase(frequencies), label=f"flower phase ({key})")
    axs[1].plot(frequencies, deep_phase(frequencies), label = f"radiant phase ({deep_key})")
    axs[1].set_xlabel("freq / GHz")
    axs[1].set_ylabel("Phase / rad")
    axs[1].legend()
    axs[1].set_title("Phase of response")
#    for ax in axs:
#        ax.set_xlim(0, 0.4)
    fig.suptitle(f"Frequency response of flower {key}")
    fig.tight_layout()
    fig.savefig("test_flower_response")

    test_dictionary = {"freq": frequencies.tolist(),
                       "flower_gain_ch2" : (565.5877570891539 * gain(frequencies)/deep_max).tolist()}
    with open("test_flower_calibration_s2023_st11.json", "w") as f:
        json.dump(test_dictionary, f)
        


    
    times = np.array(impulse_response[f"{key}_times"])
    test_times = np.arange(min(times), max(times), 1./(sampling_rate))
    response_sampled = impulse_response[key](test_times)

    fig, ax = plt.subplots()
    ax.plot(times, impulse_response[key](times), label="original")
    ax.scatter(test_times, response_sampled, color = "red", label="sampled")
    ax.set_xlim(-10, 50)
    ax.set_xlabel("time/ns")
    ax.set_ylabel("Impulse response (norm.)")
    ax.set_title("FLOWER impulse response")
    ax.legend()
    fig.savefig("test_flower_response_sampling.png")
    
    window = [-10, 30]

    selection = np.logical_and(window[0] < times, times < window[1])
    response_window = np.zeros_like(times)
    response_window[selection] = impulse_response[key](times[selection])
    response_window = interpolate.interp1d(times, response_window)
    response_window_sampled = response_window(test_times)
    
    fig, ax = plt.subplots()
    ax.plot(times, response_window(times))
    ax.set_xlim(-10, 50)
    fig.savefig("test_flower_response_window.png")
