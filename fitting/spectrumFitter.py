
import argparse
from iminuit import Minuit 
from iminuit.cost import LeastSquares
import numpy as np
from scipy.interpolate import interp1d

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import freq2time, time2freq

from utilities.utility_functions import read_pickle, find_config, read_config



def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return frequencies, frequency_spectrum, var_frequency_spectrum



class spectrumFitter:
    def __init__(self, data_path, sim_paths,
                 fit_range=[0.15, 0.6], bandpass=None):
        """
        sim_paths : list
            list of all simulation components, without the sum
            the order is assumed to be ice, electronic, galactic, rest
        """
        config_path = find_config(sim_paths[0], sim=True)
        self.config = read_config(config_path)
        
        self.channels_to_include = self.config["channels_to_include"]

        self.sampling_rate = 3.2*units.GHz
        self.frequencies, \
        self.data_spectrum, \
        self.var_data_spectrum = read_freq_spec_file(data_path)

        self.sim_spectra = []
        self.sim_var_spectra = []
        for sim_path in sim_paths:
            sim_frequencies_i, \
            sim_spectrum_i, \
            sim_var_spectrum_i = read_freq_spec_file(sim_path)
            assert np.all(self.frequencies == sim_frequencies_i)
            self.sim_spectra.append(sim_spectrum_i)
            self.sim_var_spectra.append(sim_var_spectrum_i)


        self.fit_range = fit_range
        self.fit_idxs = np.where(np.logical_and(fit_range[0] < self.frequencies,
                                                self.frequencies < fit_range[1]))

        self.bandpass = bandpass
        if self.bandpass is None:
            self.bandpass = [0.1, 0.7]

        return

    def get_fit_gain(self, mode="constant", choose_channels=None):
        if choose_channels is None:
            channel_ids = self.channels_to_include
        else:
            channel_ids = choose_channels


        fit_results = []
        for channel_id in channel_ids:
            fit_function = self.select_fit_function(mode, channel_id)
            x_data = self.frequencies[self.fit_idxs]
            y_data = self.data_spectrum[channel_id][self.fit_idxs]
            y_err = self.var_data_spectrum[channel_id][self.fit_idxs]
            cost_function = LeastSquares(x=x_data, y=y_data,
                                         yerror=y_err,
                                         model=fit_function)

            if mode == "constant":
                m = Minuit(cost_function, gain=1000.)
                m.limits = [(10, None)]
                m.migrad()
                m.hesse()
                fit_results.append(m.params)
            elif mode == "electronic_temp":
                m = Minuit(cost_function, gain=1000., el_ampl=0., el_cst=1., f0=0.15)
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True
                m.limits = [(0., None), (0., None), (0., None), (0., 0.6)]
                m.migrad()
                m.fixed["gain"] = True
                m.fixed["el_ampl"] = False
                m.fixed["el_cst"] = False
                m.fixed["f0"] = True
                m.migrad()
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True
                m.migrad()
                m.hesse()
                fit_results.append(m.params)


        return fit_results

    
    def get_fit_function(self, mode, channel_id):
        fit_function = self.select_fit_function(mode, channel_id)
        return fit_function



    def select_fit_function(self, mode, channel_id):
        ice_spectrum = self.sim_spectra[0][channel_id]
        electronic_spectrum = self.sim_spectra[1][channel_id]
        galactic_spectrum = self.sim_spectra[2][channel_id]
        if mode == "constant":
            ice_trace = freq2time(ice_spectrum, self.sampling_rate)
            electronic_trace = freq2time(electronic_spectrum, self.sampling_rate)
            galactic_trace = freq2time(galactic_spectrum, self.sampling_rate)
            def fit_gain_factor(freq, gain):
                function_interp = interp1d(self.frequencies,
                                           gain * time2freq((ice_trace
                                                           + electronic_trace
                                                           + galactic_trace),
                                            self.sampling_rate))
                return function_interp(freq)

            fit_function = fit_gain_factor
        elif mode == "electronic_temp":
            def fit_electronic_temp(freq, gain, el_ampl, el_cst, f0):
                # Johnson-Nyquist: V² ~ T

                ch_bandpass = channelBandPassFilter()
                filt = ch_bandpass.get_filter(self.frequencies, station_id=-1, channel_id=-1, det=-1, passband=self.bandpass, filter_type="butter", order=10)
                filt = np.abs(filt)
                weight = el_ampl * (self.frequencies - f0) + el_cst
                electronic_spectrum_fit = electronic_spectrum * weight * filt
                function_interp = interp1d(self.frequencies,
                                           gain * (ice_spectrum 
                                                   + electronic_spectrum_fit
                                                   + galactic_spectrum) * filt)
                
                
                return function_interp(freq)
                
            fit_function = fit_electronic_temp
        else:
            raise NotImplementedError

        return fit_function







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file")
    parser.add_argument("--sim", "-s", help="path to summed sim pickle files")
    args = parser.parse_args()


    spectrum_fitter = spectrumFitter(args.data, args.sim)
    results = spectrum_fitter.get_fit_gain()
    print(results[0])
