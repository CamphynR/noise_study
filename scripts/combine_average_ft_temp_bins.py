import argparse
import csv
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import scipy.interpolate as interpolate

from utilities.utility_functions import read_pickle, write_pickle, find_config, read_config


def read_header(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    return header


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    nr_events = header["nr_events"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, nr_events, header


def combine_mean(mean1, mean2, nr_events_1, nr_events_2):
    weighted_sum = mean1 * nr_events_1 + mean2 * nr_events_2
    return weighted_sum / (nr_events_1 + nr_events_2)


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


def create_temp_function(temp_path):
    time = []
    temperature = []
    with open(temp_path, "r") as temp_file:
        reader = csv.DictReader(temp_file)
        for i, row in enumerate(reader):
            time_tmp = float(row["time [unix s]"])
            time.append(time_tmp)
            temperature.append(float(row["heat [\N{DEGREE SIGN}C]"]))
    return interpolate.interp1d(time, temperature, bounds_error=False, fill_value=0.)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()
    args.pickles = natsorted(args.pickles)

    config_path = find_config(args.pickles[0])
    config = read_config(config_path)
    station_id = config["station"]

    nr_bins = 3
    temp_path = glob.glob(f"station_temperatures/season2023/housekeepingdata_st{station_id}_*")[0]
    temp_func = create_temp_function(temp_path)

    run_temperatures = {}
    for pickle in args.pickles:
        run_nr = os.path.basename(pickle).split(".")[0].split("run")[-1]
        header = read_header(pickle)
        begin_time = header["begin_time"].unix
        end_time = header["end_time"].unix
        begin_temp = temp_func(begin_time)
        end_temp = temp_func(end_time)
        temp = np.mean([end_temp, begin_temp])
        run_temperatures[run_nr] = temp

    hist, bin_edges = np.histogram(list(run_temperatures.values()), bins=nr_bins)
    hist_indices = np.searchsorted(bin_edges[1:-1], list(run_temperatures.values()))
    runs_per_temp = [[] for i in range(nr_bins)]
    for i, pickle in enumerate(args.pickles):
        runs_per_temp[hist_indices[i]].append(pickle)

    print(bin_edges)
    raise ValueError

    def parse_pickles(pickles, temp_range):
        frequencies_prev, frequency_spectrum_prev, var_frequency_spectrum_prev, nr_events_prev, header_prev = read_freq_spec_file(pickles[0])
        begin_time_prev, end_time_prev = header_prev["begin_time"], header_prev["end_time"]
        just_switched_month = False

        for i, pickle in enumerate(pickles[1:]):
            frequencies, frequency_spectrum, var_frequency_spectrum, nr_events, header = read_freq_spec_file(pickle)
            assert np.equal(frequencies.all(), frequencies_prev.all()), f"frequencies of {i}'th file are not equal to {i}-1th frequencies"
            var_frequency_spectrum_prev = combine_vars(var_frequency_spectrum_prev, var_frequency_spectrum,
                                                  frequency_spectrum_prev, frequency_spectrum,
                                                  nr_events_prev, nr_events)
            frequency_spectrum_prev = combine_mean(frequency_spectrum_prev, frequency_spectrum,
                                                  nr_events_prev, nr_events)
            begin_time_prev, end_time_prev = combine_times(begin_time_prev, end_time_prev, header["begin_time"], header["end_time"])
            nr_events_prev += nr_events 

        result_dictionary = read_pickle(pickles[0])
        result_dictionary["frequency_spectrum"] = frequency_spectrum_prev
        result_dictionary["var_frequency_spectrum"] = var_frequency_spectrum_prev
        print(temp_range)
        result_dictionary["header"]["nr_events"] = nr_events
        result_dictionary["header"]["temp_range"] = temp_range
                

        pickle_file = pickles[0].rsplit("_", 1)[0]
        pickle_file += f"_temp_{temp_range[0]:.0f}_{temp_range[1]:.0f}_combined.pickle"
        write_pickle(result_dictionary, pickle_file)

    for i, runs in enumerate(runs_per_temp):
        temp_range = [bin_edges[i], bin_edges[i+1]]
        parse_pickles(runs, temp_range)
