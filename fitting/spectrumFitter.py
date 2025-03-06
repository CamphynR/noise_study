
import argparse
from iminuit import Minuit 
from iminuit.cost import LeastSquares
import numpy as np

from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle, find_config, read_config



def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return frequencies, frequency_spectrum, var_frequency_spectrum



class spectrumFitter:
    def __init__(self, data_path, sim_path,
                 fit_range=[0.2, 0.6]):
        config_path = find_config(sim_path)
        self.config = read_config(config_path)
        
        self.channels_to_include = self.config["channels_to_include"]

        self.frequencies, \
        self.data_spectrum, \
        self.var_data_spectrum = read_freq_spec_file(data_path)

        sim_frequencies, \
        self.sim_spectrum, \
        self.sim_var_spectrum = read_freq_spec_file(sim_path)

        assert np.all(self.frequencies == sim_frequencies)

        self.fit_range = fit_range
        self.fit_idxs = np.where(np.logical_and(fit_range[0] < self.frequencies,
                                                self.frequencies < fit_range[1]))

        return

    def get_fit_gain(self, mode="constant"):

        fit_results = []
        for channel_id in self.channels_to_include:
            fit_function = self.select_fit_function(mode, channel_id)

            x_data = self.frequencies[self.fit_idxs]
            y_data = self.data_spectrum[channel_id][self.fit_idxs]
            y_err = self.var_data_spectrum[channel_id][self.fit_idxs]

            cost_function = LeastSquares(x=x_data, y=y_data,
                                         yerror=y_err,
                                         model=fit_function)

            m = Minuit(cost_function, gain=1.)
            m.migrad()
            m.hesse()
            fit_results.append(m.params)

        return fit_results



    def select_fit_function(self, mode, channel_id):
        if mode == "constant":
            def fit_gain_factor(freq, gain):
                spectrum = self.sim_spectrum[channel_id][self.fit_idxs]
                index = np.nonzero(freq == self.frequencies[self.fit_idxs])
                return gain * np.squeeze(spectrum[index])
            fit_function = fit_gain_factor
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
