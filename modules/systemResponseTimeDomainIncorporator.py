import argparse
import glob
import json
import numpy as np
import os
from scipy import interpolate


from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

def open_response_file(response_path):
    with open(response_path, "r") as response_file:
        response_dic = json.load(response_file)
    times = response_dic["time"]

    keys = list(response_dic.keys())
    for key in keys:
        if key == "time":
            continue
        else:
            channel_times_key = f"{key}_times"
            cur_times = np.array(times) - times[np.argmax(np.abs(response_dic[key]))]
            ch_response = response_dic[key]
            # ch_response = np.pad(2048, ch_response)
            response_interpol = interpolate.interp1d(cur_times, ch_response,
                                                     kind="linear", bounds_error=False, fill_value=0)
            response_dic[channel_times_key] = cur_times
            response_dic[key] = response_interpol
    return response_dic


def convert_response_to_spectrum(response_dic, response_channel_key, window = [-10, 30]):
    times = response_dic[f"{response_channel_key}_times"]
    response = response_dic[response_channel_key]
    response = response(times)
    selection = np.logical_not(np.logical_and(window[0] < times, times < window[1]))
    response[selection] = 0
    sampling_rate = 1./np.diff(times[:2])[0]
    frequencies = np.fft.rfftfreq(len(times), d=1./sampling_rate)
    spectrum = time2freq(response, sampling_rate)
    spectrum_interpol = interpolate.interp1d(frequencies, spectrum,
                                             bounds_error=False, fill_value=0+0j)
    return frequencies, spectrum_interpol


def convert_response_to_time_domain(response, nr_samples=2048, sampling_rate=3.2*units.GHz, debug=False):
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    selection = np.where(np.logical_and(0.1 < frequencies, frequencies < 0.8))
    bandpass = np.zeros_like(frequencies)
    bandpass[selection] = 1.
    response_sampled = response(frequencies) * bandpass
    if debug:
        import matplotlib.pyplot as plt
        plt.close()
        plt.plot(frequencies, np.abs(response_sampled))
        plt.show()
    times = np.arange(-1*nr_samples/2, nr_samples/2) / sampling_rate
    response_sampled[0] = 0
    impulse_response = freq2time(response_sampled, sampling_rate)
    impulse_response_max_idx = np.argmax(np.abs(impulse_response))
    impulse_response = np.roll(impulse_response, -1*impulse_response_max_idx + int(nr_samples/2))
    impulse_response_max_idx = np.argmax(np.abs(impulse_response))
    cur_times = np.array(times) - times[impulse_response_max_idx]
    impulse_response = impulse_response / np.max(impulse_response)
    if debug:
        plt.close()
        plt.plot(cur_times, impulse_response)
        plt.xlim(-10, 30)
        plt.show()
        plt.close()
    impulse_response = interpolate.interp1d(cur_times, impulse_response)
    return cur_times, impulse_response



class systemResonseTimeDomainIncorporator():
    def __init__(self):
        self.channel_mapping = {
            "ch2_6dB" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
            "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
            }
        self.channel_mapping = {i : key for key, value in zip(self.channel_mapping.keys(), self.channel_mapping.values()) for i in value}
        return


    def begin(self, response_path, det):
        self.response = open_response_file(response_path)
        surface_response = det.get_component(collection="full_chain", component="impulse_response_st13_ch19")
        surface_times, surface_response = convert_response_to_time_domain(surface_response)
        self.response["surface_times"] = surface_times
        self.response["surface"] = surface_response


    def run(self, event, station, det):
        for channel in station.iter_channels():
            sampling_rate = channel.get_sampling_rate()
            trace_with_response = self.apply_response(channel, self.response)
            channel.set_trace(trace_with_response, sampling_rate)
            # spectrum_with_response = self.apply_response(channel, self.response_spectrum)
            # channel.set_frequency_spectrum(spectrum_with_response, sampling_rate)


    def apply_response(self, channel, response_dic, window=[-10, 30]):
        channel_id = channel.get_id()
        response_key = self.channel_mapping[channel_id]
        times = response_dic[f"{response_key}_times"]
        response = response_dic[response_key]
        sampling_rate = channel.get_sampling_rate()
        sample_times = np.arange(min(times), max(times), 1./sampling_rate)
        response_sampled = response(sample_times)
        trace_with_response = np.convolve(channel.get_trace(), response_sampled, mode="same") 

        selection = np.logical_and(window[0] < times, times < window[1])
        response_window = np.zeros_like(times)
        response_window[selection] = response(times[selection])

        sample_diff = len(trace_with_response) - len(channel.get_trace())
        trace_with_response = trace_with_response[int(sample_diff/2):int(len(trace_with_response) - sample_diff/2)]
        assert len(trace_with_response) == len(channel.get_trace())
        return trace_with_response

    
    def get_response(self, channel_id=None, response_channel_key=None):
        if channel_id is not None:
            response_key = self.channel_mapping[channel_id]
        elif response_channel_key is not None:
            response_key = response_channel_key
        else:
            raise ValueError("you should provide either a channel_id or response key")
        
        times = self.response[f"{response_key}_times"]
        response = self.response[response_key]

        return times, response 



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", default=23)
    parser.add_argument("--run", "-r", default=101)
    args = parser.parse_args()

    try:
        plt.style.use("gaudi")
    except:
        pass

    det = 0
    response_path = "/home/ruben/Documents/projects/RNO-G_noise_study/sim/library/deep_impulse_responses.json"

    response_dic = open_response_file(response_path)
    response = response_dic["ch2"]

    
    times = np.array(response_dic["ch2_times"])
    test_times = np.arange(min(times), max(times), 1./(3.2*units.GHz))
    response_sampled = response(test_times)

    plt.plot(times, response(times), label="original")
    plt.scatter(test_times, response_sampled, color = "red", label="sampled")
    plt.xlim(-10, 50)
    plt.xlabel("time/ns")
    plt.ylabel("Impulse response (norm.)")
    plt.title("Impulse response")
    plt.legend()
    plt.show()


    data_dir = os.environ["RNO_G_DATA"]
    root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}")

    reader = readRNOGData()
    mattak_kw = {"cache_calibration" : False}
    reader.begin(root_dirs,
                 mattak_kwargs=mattak_kw)


    system_incorporator = systemResonseTimeDomainIncorporator()
    system_incorporator.begin(response_path)


    for event in reader.run():
        station = event.get_station()
        for channel in station.iter_channels():
            times = channel.get_times()
            plt.plot(times, channel.get_trace(), label="before response")
            break

        system_incorporator.run(event, station, det)

        for channel in station.iter_channels():
            times = channel.get_times()
            plt.plot(times, channel.get_trace(), label="after response", zorder=-1)
            break
        plt.xlabel("Time / ns")
        plt.ylabel("Amplitude / V")
        plt.title("trace channel 0")
        plt.legend()
        plt.show()
        break
