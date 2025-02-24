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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file")
    parser.add_argument("--sim", "-s", help="path to summed sim pickle files")
    args = parser.parse_args()

    frequencies, data_spectrum, var_data_spectrum = read_freq_spec_file(args.data)
    sum_frequencies, sum_spectrum, _ = read_freq_spec_file(args.sim)

    assert np.all(sum_frequencies == frequencies)

    channel_id = 0
    x_data = frequencies
    y_data = data_spectrum[channel_id]
    y_err = var_data_spectrum[channel_id]

    def fit_gain_factor(freq, gain):
        spectrum = sum_spectrum[channel_id]
        index = np.nonzero(freq == frequencies)
        return gain * np.squeeze(spectrum[index])

    fit_function = fit_gain_factor
    cost_function = LeastSquares(x=x_data, y=y_data,
                                 yerror=y_err,
                                 model=fit_function)

    m = Minuit(cost_function, gain=1.)
    m.migrad()
    m.hesse()
    print(m.params)
