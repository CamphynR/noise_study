import argparse
import numpy as np

from utilities.utility_functions import read_pickle, write_pickle


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    nr_events = header["nr_events"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, nr_events

def combine_mean(mean1, mean2, nr_events_1, nr_events_2):
    weighted_sum = mean1 * nr_events_1 + mean2 * nr_events_2
    return weighted_sum + (nr_events_1 + nr_events_2)


def combine_vars(var1, var2, mean1, mean2, nr_events_1, nr_events_2):
    weighted_sum = (nr_events_1 - 1) * var1 + (nr_events_2 * var2)
    correction_factor = nr_events_1 * nr_events_2 * (mean1 - mean2)**2

    return weighted_sum / (nr_events_1 + nr_events_2 - 1) + correction_factor / ((nr_events_1 + nr_events_2) * (nr_events_1 + nr_events_2 - 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()
    
    frequencies_prev, frequency_spectrum_prev, var_frequency_spectrum_prev, nr_events_prev = read_freq_spec_file(args.pickles[0])
    for i, pickle in enumerate(args.pickles[1:]):
        frequencies, frequency_spectrum, var_frequency_spectrum, nr_events = read_freq_spec_file(pickle)
        assert np.equal(frequencies.all(), frequencies_prev.all()), f"frequencies of {i}'th file are not equal to {i}-1th frequencies"
        var_frequency_spectrum = combine_vars(var_frequency_spectrum_prev, var_frequency_spectrum,
                                              frequency_spectrum_prev, frequency_spectrum,
                                              nr_events_prev, nr_events)
        frequency_spectrum = combined_freq_spectrum = combine_mean(frequency_spectrum_prev, frequency_spectrum,
                                              nr_events_prev, nr_events)
        nr_events_prev += nr_events 

    result_dictionary = read_pickle(args.pickles[0])
    result_dictionary["frequency_spectrum"] = frequency_spectrum
    result_dictionary["var_frequency_spectrum"] = var_frequency_spectrum
    result_dictionary["header"]["nr_events"] = nr_events

    pickle_file = args.pickles[0].rsplit("_", 1)[0]
    pickle_file += "_combined.pickle"
    write_pickle(result_dictionary, pickle_file)
