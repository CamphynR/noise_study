import argparse
import glob
import json
import numpy as np
import os
from scipy import interpolate


from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units

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
            print(channel_times_key)
            cur_times = np.array(times) - times[np.argmax(np.abs(response_dic[key]))]
            ch_response = response_dic[key]
            response_interpol = interpolate.interp1d(cur_times, ch_response,
                                                     kind="linear", bounds_error=False, fill_value=0)
            response_dic[channel_times_key] = cur_times
            response_dic[key] = response_interpol

    return response_dic



class systemResonseTimeDomainIncorporator():
    def __init__(self):
        # mapping to map channels to the keys in the json file
        self.channel_mapping = {"ch2" : [0, 1, 2, 3],
                                "ch11_6dB" : [4, 8, 11, 21],
                                "ch9_6dB" : [9, 10, 22, 23]}
        self.channel_mapping = {key: group for group, keys in self.channel_mapping.items() for key in keys}
        return


    def begin(self, response_path):
        self.response = open_response_file(response_path)


    def run(self, event, station, det):
        for channel in station.iter_channels():
            sampling_rate = channel.get_sampling_rate()
            trace_with_response = self.apply_response(channel, self.response)
            channel.set_trace(trace_with_response, sampling_rate)


    def apply_response(self, channel, response_dic):
        channel_id = channel.get_id()
        if channel_id in self.channel_mapping.keys():
            response_key = self.channel_mapping[channel_id]
        else:
            response_key = "ch2"

        sampling_rate = channel.get_sampling_rate()

        times = response_dic[f"{response_key}_times"]
        response = response_dic[response_key]
        sample_times = np.arange(min(times), max(times), 1./sampling_rate)
        response_sampled = response(sample_times)

        trace_with_response = np.convolve(channel.get_trace(), response_sampled, mode="same") 

        sample_diff = len(trace_with_response) - len(channel.get_trace())
        trace_with_response = trace_with_response[int(sample_diff/2):int(len(trace_with_response) - sample_diff/2)]
        print(len(trace_with_response))
        assert len(trace_with_response) == len(channel.get_trace())
        return trace_with_response

    
    def get_response(self, channel_id):
        if channel_id in self.channel_mapping.keys():
            response_key = self.channel_mapping[channel_id]
        else:
            response_key = "ch2"
        
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
