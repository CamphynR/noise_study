import argparse
from natsort import natsorted
import numpy as np

from utilities.utility_functions import read_pickle, write_pickle


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    nr_events = header["nr_events"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, nr_events, header

def combine_mean(mean1, mean2, nr_events_1, nr_events_2):
    weighted_sum = mean1**2 * nr_events_1 + mean2**2 * nr_events_2
    return np.sqrt(weighted_sum / (nr_events_1 + nr_events_2))

def combine_times(begin_time1, end_time1, begin_time2, end_time2):
    if begin_time1 < begin_time2:
        begin_time = begin_time1
    else:
        begin_time = begin_time2
    if end_time1 > end_time2:
        end_time = end_time1
    else:
        end_time = end_time2
    return begin_time, end_time


def combine_vars(var1, var2, mean1, mean2, nr_events_1, nr_events_2):
    weighted_sum = (nr_events_1 - 1) * var1 + (nr_events_2 * var2)
    correction_factor = nr_events_1 * nr_events_2 * (mean1 - mean2)**2

    return weighted_sum / (nr_events_1 + nr_events_2 - 1) + correction_factor / ((nr_events_1 + nr_events_2) * (nr_events_1 + nr_events_2 - 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()
    
    args.pickles = natsorted(args.pickles)

    frequencies_prev, frequency_spectrum_prev, var_frequency_spectrum_prev, nr_events_prev, header_prev = read_freq_spec_file(args.pickles[0])
    begin_time_prev, end_time_prev = header_prev["begin_time"], header_prev["end_time"]
    just_switched_month = False

    for i, pickle in enumerate(args.pickles[1:]):
        frequencies, frequency_spectrum, var_frequency_spectrum, nr_events, header = read_freq_spec_file(pickle)
        if just_switched_month:
            frequency_spectrum_prev = frequency_spectrum
            var_frequency_spectrum_prev = var_frequency_spectrum
            nr_events = 0
            begin_time_prev = header["begin_time"]
            end_time_prev = header["end_time"]
            just_switched_month = False
            continue

        assert np.equal(frequencies.all(), frequencies_prev.all()), f"frequencies of {i}'th file are not equal to {i}-1th frequencies"
        var_frequency_spectrum_prev = combine_vars(var_frequency_spectrum_prev, var_frequency_spectrum,
                                              frequency_spectrum_prev, frequency_spectrum,
                                              nr_events_prev, nr_events)
        frequency_spectrum_prev = combine_mean(frequency_spectrum_prev, frequency_spectrum,
                                              nr_events_prev, nr_events)
        begin_time_prev, end_time_prev = combine_times(begin_time_prev, end_time_prev, header["begin_time"], header["end_time"])
        nr_events_prev += nr_events 

        prev_month = header_prev["begin_time"].datetime.month
        month = header["begin_time"].datetime.month
        if prev_month != month:
            print("New month, saving \n ----------")
            result_dictionary = read_pickle(args.pickles[0])
            result_dictionary["frequency_spectrum"] = frequency_spectrum_prev
            result_dictionary["var_frequency_spectrum"] = var_frequency_spectrum_prev
            print(begin_time_prev)
            print(end_time_prev)
            result_dictionary["header"]["nr_events"] = nr_events
            result_dictionary["header"]["begin_time"] = begin_time_prev
            result_dictionary["header"]["end_time"] = end_time_prev
            
            just_switched_month = True

            pickle_file = args.pickles[0].rsplit("_", 1)[0]
            pickle_file += f"_month_{prev_month}_combined.pickle"
            write_pickle(result_dictionary, pickle_file)


        header_prev = header


    print("New month, saving \n ----------")
    result_dictionary = read_pickle(args.pickles[0])
    result_dictionary["frequency_spectrum"] = frequency_spectrum_prev
    result_dictionary["var_frequency_spectrum"] = var_frequency_spectrum_prev
    print(begin_time_prev)
    print(end_time_prev)
    result_dictionary["header"]["nr_events"] = nr_events
    result_dictionary["header"]["begin_time"] = begin_time_prev
    result_dictionary["header"]["end_time"] = end_time_prev
    
    just_switched_month = True

    pickle_file = args.pickles[0].rsplit("_", 1)[0]
    pickle_file += f"_month_{prev_month}_combined.pickle"
    write_pickle(result_dictionary, pickle_file)
