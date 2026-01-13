import argparse
import copy
import glob
import json
import numpy as np
import os
from scipy import interpolate
import scipy.constants as constants
from scipy.signal.windows import tukey


from NuRadioReco.detector.RNO_G.analog_components import load_amp_response
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

"""
REMEMBER ALL SAMPLING RATE DEFAULTS ARE SET TO 3.2 GHz!!! (which is wrong for 2024 RADIANT v3)
"""


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


def rescale_response(response, integration_range = [0.08*units.GHz, 0.8*units.GHz],
                     nr_samples=2048, sampling_rate=3.2*units.GHz,
                     return_norm=False):
    """
    Function to rescale the given response such that it's max is set to 1
    """
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
#    f0 = sampling_rate/nr_samples
#    i80 = np.where(np.isclose(frequencies, 80 * units.MHz, atol=0.5*f0))[0][0]
#    i800 = np.where(np.isclose(frequencies, 800 * units.MHz, atol=0.5*f0))[0][0]
#    energy = np.trapezoid(response(frequencies)[i80:i800]**2)
    max_response = np.max(response(frequencies))

    new_response = interpolate.interp1d(frequencies, response(frequencies)/max_response)

    if return_norm:
        return new_response, max_response
    return new_response
    


def convert_response_to_spectrum(response_dic, response_channel_key, window = [-15, 40], bandpass_kwargs=None):
    response_dic_temp = copy.copy(response_dic)
    times = response_dic_temp[f"{response_channel_key}_times"]
    response = response_dic_temp[response_channel_key]
    response = response(times)

    selection = np.logical_not(np.logical_and(window[0] < times, times < window[1]))
    window = tukey(len(response[np.invert(selection)]))
    response[selection] = 0
    response[np.invert(selection)] *= window
    sampling_rate = 1./np.diff(times[:2])[0]
    frequencies = np.fft.rfftfreq(len(times), d=1./sampling_rate)
    spectrum = time2freq(response, sampling_rate)

    if bandpass_kwargs is not None:
        bandpas_filter = channelBandPassFilter()
        # 0 0 0 just fills in the keywords station, channel, det.
        # these are not used in the function but included to follow NuRadio's structure
        bandpas_filter = bandpas_filter.get_filter(frequencies, 0, 0, 0,
                                                   **bandpass_kwargs)
        spectrum *= bandpas_filter

    gain_interpol = interpolate.interp1d(frequencies, np.abs(spectrum),
                                             bounds_error=False, fill_value=0+0j)
    phase_interpol = interpolate.interp1d(frequencies, np.unwrap(np.angle(spectrum), period=np.pi),
                                             bounds_error=False, fill_value=0+0j)
    return gain_interpol, phase_interpol


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



def temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance=50*units.ohm, filter_type="rectangular"):
    if filter_type=="rectangular":
        filt = np.zeros_like(frequencies)
        filt[np.where(np.logical_and(min_freq < frequencies , frequencies < max_freq))] = 1
    else:
        print("Other filters not yet implemented")
    bandwidth = np.trapezoid(np.abs(filt)**2, frequencies)
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    vrms = np.sqrt(k * temperature * resistance * bandwidth)
    return vrms



class systemResponseTimeDomainIncorporator():
    def __init__(self):
        self.channel_mapping_inv = {
            "deep" : [0, 1, 2, 3],
            "helper" : [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
            "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
            }
        self.channel_mapping = {i : key for key, value in zip(self.channel_mapping_inv.keys(), self.channel_mapping_inv.values()) for i in value}
        return


    def begin(self, det, response_path=None, normalize=True, overwrite_key=None, bandpass_kwargs=None):
        """
        only_response_path: the module will not try to make responses for deep helper and surface but will only load the given response path
        """

        if bandpass_kwargs is not None and not isinstance(bandpass_kwargs, dict):
            raise TypeError("bandpass only accepts a dict of kwargs for the module channelBandPassFilter's function get_filter")

        self.response = {key : {} for key in self.channel_mapping_inv.keys()}
        self.normalizations = {key : {} for key in self.channel_mapping_inv.keys()}
        if response_path is None:
            self.response["deep"] = load_amp_response(amp_type="deep_impulse")
            self.response["helper"] = copy.copy(self.response["deep"])
            self.response["surface"] = load_amp_response(amp_type="rno_surface_impulse")

            self.response["deep"]["gain"] = rescale_response(self.response["deep"]["gain"])
            self.response["helper"]["gain"] = rescale_response(self.response["helper"]["gain"])
            self.response["surface"]["gain"] = rescale_response(self.response["surface"]["gain"])
            return

        if len(response_path) == 2:
            impulse_response_deep = open_response_file(response_path[0])
            impulse_response_surface = open_response_file(response_path[1])
            
            if isinstance(overwrite_key, dict):
                deep_key = overwrite_key["deep"]
                helper_key = overwrite_key["helper"]
                surface_key = overwrite_key["surface"]

                self.response["deep"]["gain"], self.response["deep"]["phase"] = convert_response_to_spectrum(impulse_response_deep, deep_key, bandpass_kwargs=bandpass_kwargs)
                self.response["helper"]["gain"], self.response["helper"]["phase"]= convert_response_to_spectrum(impulse_response_deep, helper_key, bandpass_kwargs=bandpass_kwargs)
                self.response["surface"]["gain"], self.response["surface"]["phase"] = convert_response_to_spectrum(impulse_response_surface, surface_key, bandpass_kwargs=bandpass_kwargs)
            elif overwrite_key:
                deep_key = overwrite_key
                helper_key = overwrite_key
                surface_key = overwrite_key

                if overwrite_key in impulse_response_deep.keys():
                    self.response["deep"]["gain"], self.response["deep"]["phase"] = convert_response_to_spectrum(impulse_response_deep, deep_key, bandpass_kwargs=bandpass_kwargs)
                    self.response["helper"]["gain"], self.response["helper"]["phase"]= convert_response_to_spectrum(impulse_response_deep, helper_key, bandpass_kwargs=bandpass_kwargs)
                    self.response["surface"]["gain"], self.response["surface"]["phase"] = convert_response_to_spectrum(impulse_response_deep, surface_key, bandpass_kwargs=bandpass_kwargs)
                elif overwrite_key in impulse_response_surface.keys():
                    self.response["deep"]["gain"], self.response["deep"]["phase"] = convert_response_to_spectrum(impulse_response_surface, deep_key, bandpass_kwargs=bandpass_kwargs)
                    self.response["helper"]["gain"], self.response["helper"]["phase"]= convert_response_to_spectrum(impulse_response_surface, helper_key, bandpass_kwargs=bandpass_kwargs)
                    self.response["surface"]["gain"], self.response["surface"]["phase"] = convert_response_to_spectrum(impulse_response_surface, surface_key, bandpass_kwargs=bandpass_kwargs)
                elif overwrite_key == "surface_query":
                    response_tmp = load_amp_response(amp_type="rno_surface_impulse")
                    response_tmp_phase = response_tmp["phase"]
                    # we choose this fine enough to avoid any artifacts, the responses at this point
                    # do not yet correspond to a radiant versions frequencies
                    tmp_freqs = np.linspace(0., 1.6, 10000)
                    response_tmp_phase = response_tmp_phase(tmp_freqs)
                    response_tmp_phase = np.unwrap(np.angle(response_tmp_phase), period=np.pi)
                    response_tmp_phase = interpolate.interp1d(tmp_freqs, response_tmp_phase,
                                                              bounds_error=False, fill_value=0+0j)
                    response_tmp_gain = rescale_response(response_tmp["gain"])
                    if bandpass_kwargs is not None:
                        bandpas_filter = channelBandPassFilter()
                        # 0 0 0 just fills in the keywords station, channel, det.
                        # these are not used in the function but included to follow NuRadio's structure
                        freqs = np.arange(0, 1.6, 0.01)
                        bandpas_filter = bandpas_filter.get_filter(freqs, 0, 0, 0,
                                                                   **bandpass_kwargs)
                        response_tmp_gain = interpolate.interp1d(
                                freqs,
                                np.abs(bandpas_filter) * response_tmp_gain(freqs),
                                bounds_error=False,
                                fill_value=0.)
                    self.response["deep"]["gain"], self.response["deep"]["phase"] = response_tmp_gain, response_tmp_phase
                    self.response["helper"]["gain"], self.response["helper"]["phase"]= response_tmp_gain, response_tmp_phase
                    self.response["surface"]["gain"], self.response["surface"]["phase"] = response_tmp_gain, response_tmp_phase


            else:
                deep_key = "v3_ch1_62dB"
                helper_key = "v3_ch4_62dB"
                surface_key = "v3_ch13"

                self.response["deep"]["gain"], self.response["deep"]["phase"] = convert_response_to_spectrum(impulse_response_deep, deep_key, bandpass_kwargs=bandpass_kwargs)
                self.response["helper"]["gain"], self.response["helper"]["phase"]= convert_response_to_spectrum(impulse_response_deep, helper_key, bandpass_kwargs=bandpass_kwargs)
                self.response["surface"]["gain"], self.response["surface"]["phase"] = convert_response_to_spectrum(impulse_response_surface, surface_key, bandpass_kwargs=bandpass_kwargs)
                
                
        else:
            # assume only deep response is given since for 2023
            # surface response is not available as a json so this is still queried
            impulse_response_deep = open_response_file(response_path)
            self.response["surface"] = load_amp_response(amp_type="rno_surface_impulse")
            
            self.response["deep"]["gain"], self.response["deep"]["phase"] = convert_response_to_spectrum(impulse_response_deep, "ch2", bandpass_kwargs=bandpass_kwargs)
            try:
                self.response["helper"]["gain"], self.response["helper"]["phase"] = convert_response_to_spectrum(impulse_response_deep, "ch9_6dB", bandpass_kwargs=bandpass_kwargs)
                self.response["helper"]["gain"] = rescale_response(self.response["helper"]["gain"])
            except:
                print("no helper channels found")




        if normalize:
            self.response["deep"]["gain"], self.normalizations["deep"] = rescale_response(self.response["deep"]["gain"], return_norm=True)
            self.response["surface"]["gain"], self.normalizations["surface"] = rescale_response(self.response["surface"]["gain"], return_norm=True)

        return


    def run(self, event, station, det):
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            response_key = self.channel_mapping[channel_id]
            response = self.response[response_key]["gain"]

            freqs = channel.get_frequencies()
            spectrum = channel.get_frequency_spectrum()

            spectrum_with_response = self.apply_response(spectrum, freqs, response)
            channel.set_frequency_spectrum(spectrum_with_response, sampling_rate="same")
        return

    def apply_response(self, spectrum, frequencies, response):
        spectrum_with_response = spectrum * response(frequencies)
        return spectrum_with_response

    def get_response(self, channel_id):
        response_key = self.channel_mapping[channel_id] 
        response = self.response[response_key]
        return response

    def get_normalization(self, channel_id):
        response_key = self.channel_mapping[channel_id] 
        norm = self.normalizations[response_key]
        return norm

    def save_response(self, filename, nr_samples=2048, sampling_rate=3.2*units.GHz,
                      channel_id=None):
        """
        channel_id : None or int
            If specified only save the response for the given channel id,
            the json dict will have keys freq, gain, phase
            If None, save a full station template,
            the json dict will have keys gain_deep, gain_helper, gain_surface and likewise for the phase
        """
        frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate).tolist()
        json_dic = {}
        json_dic["frequencies"] = frequencies
        if channel_id is None:
            json_dic["gain_deep"] = self.response["deep"]["gain"](frequencies).tolist()
            json_dic["gain_helper"] = self.response["helper"]["gain"](frequencies).tolist()
            json_dic["gain_surface"] = self.response["surface"]["gain"](frequencies).tolist()

            json_dic["phase_deep"] = self.response["deep"]["phase"](frequencies).tolist()
            json_dic["phase_helper"] = self.response["helper"]["phase"](frequencies).tolist()
            json_dic["phase_surface"] = np.angle(self.response["surface"]["phase"](frequencies)).tolist()
        else:
            response_key = self.channel_mapping[channel_id]
            response = self.response[response_key]
            json_dic["gain"] = response["gain"](frequencies).tolist()
            json_dic["phase"] = response["phase"](frequencies).tolist()



        with open(filename, "w") as file:
            json.dump(json_dic, file)
        return json_dic
        
        


#    def apply_response(self, channel, response_dic, window=[-10, 30]):
#        channel_id = channel.get_id()
#        response_key = self.channel_mapping[channel_id]
#        times = response_dic[f"{response_key}_times"]
#        response = response_dic[response_key]
#
#        selection = np.logical_and(window[0] < times, times < window[1])
#        response_window = np.zeros_like(times)
#        response_window[selection] = response(times[selection])
#        response_window = interpolate.interp1d(times, response_window)
#
#        sampling_rate = channel.get_sampling_rate()
#        sample_times = np.arange(min(times), max(times), 1./sampling_rate)
#        response_sampled = response_window(sample_times)
#        trace_with_response = np.convolve(channel.get_trace(), response_sampled, mode="same") 
#
#        sample_diff = len(trace_with_response) - len(channel.get_trace())
#        trace_with_response = trace_with_response[int(sample_diff/2):int(len(trace_with_response) - sample_diff/2)]
#        assert len(trace_with_response) == len(channel.get_trace())
#        return trace_with_response
#
#    
#    def get_response(self, channel_id=None, response_channel_key=None):
#        if channel_id is not None:
#            response_key = self.channel_mapping[channel_id]
#        elif response_channel_key is not None:
#            response_key = response_channel_key
#        else:
#            raise ValueError("you should provide either a channel_id or response key")
#        
#        times = self.response[f"{response_key}_times"]
#        response = self.response[response_key]
#
#        return times, response 



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder


    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", default=23)
    parser.add_argument("--channel", "-c", default=0, type=int)
    parser.add_argument("--run", "-r", default=101)
    parser.add_argument("--save_response", action="store_true")
    args = parser.parse_args()

    response_path = "/user/rcamphyn/noise_study/sim/library/deep_impulse_responses.json"
#    response_path = "/user/rcamphyn/noise_study/sim/library/flower_impulse_responses.json"
    
    freq_range = [0., 1.6]
    nr_samples = 2048
    sampling_rate= 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    noise_adder = channelGenericNoiseAdder()
    noise_adder.begin()
    amplitude = temp_to_volt(80 * units.kelvin, freq_range[0], freq_range[1], frequencies)


    passband = [0.1, 0.7]
    bandpass_kwargs = dict(passband=passband, filter_type="butter", order=10) 

    

    system_incorporator = systemResponseTimeDomainIncorporator()
    system_incorporator.begin(det=0,
                              response_path=response_path,
                              bandpass_kwargs=bandpass_kwargs
                              )

    if args.save_response:
        system_incorporator.save_response("system_response.json")

    response = system_incorporator.get_response(args.channel)
    phase = response["phase"]
    response = response["gain"]
    nr_sims = 1000
    test_noise = []
    for i in range(nr_sims): 
        test_noise_i = noise_adder.bandlimited_noise(*freq_range, nr_samples, sampling_rate, amplitude,
                                                     type="rayleigh", time_domain=False)
        test_noise_i = system_incorporator.apply_response(test_noise_i, frequencies, response)
        test_noise.append(test_noise_i)
    test_noise = np.mean(np.abs(test_noise), axis=0)


    plt.style.use("gaudi")
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0][0].plot(frequencies, response(frequencies))
    axs[0][0].set_ylabel("Gain")
    axs[0][0].set_title("Gain of respone")
    axs[0][1].plot(frequencies, phase(frequencies))
    axs[0][1].set_ylabel("Phase / rad")
    axs[0][1].set_title("Phase of respone")
    axs[1][0].plot(frequencies, test_noise)
    axs[1][0].set_xlim(0, 1.)
    axs[1][0].set_xlabel("freq / GHz")
    axs[1][0].set_ylabel("Spectral amplitude / V/GHz")
    axs[1][0].set_title(f"mean of {nr_sims} generic noise spectra")
    fig.tight_layout()
    fig.savefig("test_system_response")

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
    plt.savefig("test_response_sampling.png")
    plt.close()
    
    window = [-10, 30]

    selection = np.logical_and(window[0] < times, times < window[1])
    response_window = np.zeros_like(times)
    response_window[selection] = response(times[selection])
    response_window = interpolate.interp1d(times, response_window)
    response_window_sampled = response_window(test_times)
    
    plt.plot(times, response_window(times))
    plt.xlim(-10, 50)
    plt.savefig("test_response_window.png")
