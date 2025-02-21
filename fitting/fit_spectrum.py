import argparse
from iminuit import Minuit, LeastSquares
import numpy as np

from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    return frequencies, frequency_spectrum


def fit_gain_factor(ice_spec, el_spec, gal_spec, gain):
    return gain * (ice_spec + el_spec + gal_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file")
    parser.add_argument("--ice_sim", "-ice", help="path to ice pickle files")
    parser.add_argument("--electronic_sim", "-el", help="path to electronic pickle files")
    parser.add_argument("--galactic_sim", "-gal", help="path to galactic pickle files")
    args = parser.parse_args()

    data_dic = read_freq_spec_file(args.data)
    ice_dic = read_freq_spec_file(args.ice_sim)
    el_dic = read_freq_spec_file(args.electronic_sim)
    gal_dic = read_freq_spec_file(args.galactic_sim)

    frequencies = data_dic["frequencies"]
    assert np.all([dic["frequencies"]==frequencies for dic in [ice_dic, el_dic, gal_dic]])


    x_data = np.array([ice_dic["frequency_spectrum"],
                       el_dic["frequency_spectrum"],
                       gal_dic["frequency_spectrum"]
                       ])

    y_data = np.array([data_dic["frequency_spectrum"]])


    fit_function = fit_gain_factor
    cost_function = LeastSquares(x_data, y_data, fit_function)

    m = Minuit(cost_function, gain=1.)
    m.migrad()
