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

    fit_range = [200*units.MHz, 600 * units.MHz]
    fit_idxs = np.where(np.logical_and(fit_range[0] < frequencies, frequencies < fit_range[1]))

    channel_id = 4
    x_data = frequencies[fit_idxs]
    y_data = data_spectrum[channel_id][fit_idxs]
    y_err = var_data_spectrum[channel_id][fit_idxs]

    def fit_gain_factor(freq, gain):
        spectrum = sum_spectrum[channel_id][fit_idxs]
        index = np.nonzero(freq == x_data)
        return gain * np.squeeze(spectrum[index])

    fit_function = fit_gain_factor
    cost_function = LeastSquares(x=x_data, y=y_data,
                                 yerror=y_err,
                                 model=fit_function)

    m = Minuit(cost_function, gain=1.)
    m.migrad()
    m.hesse()
    print(m.params)
