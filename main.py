"""
This file is written to process RNOG data using the NuRadio modules, in particular the readRNOGDataMattak module.
One can choose cleaning modules (either homemade or directly frim NuRadio) and a variable to process and the script will
run over all available data, calculate the variable and store it in a pickle file.

Whether all variables or only the mean over all events is to be stored can be chosen in the argument parser using the --only_mean flag

The whole script uses NuRadio base units (freq in GHz, time in ns)
"""

import os
import time
import logging
import glob
import json
import pickle
import argparse
import datetime
import subprocess
import tracemalloc
from typing import Callable

from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from astropy.time import Time
import NuRadioReco
import NuRadioReco.modules
import NuRadioReco.modules.RNO_G
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.io.eventReader import eventReader
#from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator

import modules.cwFilter
from main_parser_functions import parse_data, functions

logging.basicConfig(level = logging.WARNING)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_output_shape(function : Callable[[BaseTrace], np.ndarray]):
    """
    Function to obtain the output shape of a function that applies to a NuRadio BaseTrace object
    """
    dummy_channel = BaseTrace()
    dummy_trace = np.ones(2048)
    dummy_fs = 3.2e9 * units.Hz
    dummy_channel.set_trace(dummy_trace, sampling_rate = dummy_fs)
    dummy_output = function(dummy_channel)
    # check to get shape (1) of output is a single entity e.g. an int or float
    if np.array(dummy_output).shape:
        shape = np.array(dummy_output).shape
    else:
        shape = (1,)
    return shape

def initialise_variables_list(function, nr_of_channels = 24):
    output_shape = get_output_shape(function)
    list_shape = tuple([nr_of_channels] + list(output_shape))
    variables_list = np.zeros(list_shape)
    return variables_list

@jit
def rms_numba(trace):
    return np.sqrt(np.mean(trace**2))

def calculate_rms(channel):
    trace = channel.get_trace()
    rms = rms_numba(trace)
    return rms

def calculate_trace(channel):
    return channel.get_trace()

def calculate_spec(channel):
    spec = channel.get_frequency_spectrum()
    return np.abs(spec)

def calculate_spec_hist(channel, hist_range, nr_bins):
    sampling_rate = channel.get_sampling_rate()
    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()
    spec = np.abs(spec)

    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    # digitize takes INNER edges
    bin_idxs = np.digitize(spec, bin_edges[1:-1])
    return freqs, bin_idxs, sampling_rate
    

def calculate_trace_hist(channel, hist_range, nr_bins):
    bins = np.linspace(hist_range[0], hist_range[1], nr_bins)
    trace = channel.get_trace()
    hist, edges = np.histogram(trace, bins=bins)
    return hist, edges

def select_config(config_value : str, options : dict) -> np.ndarray:
    for option in options.keys():
        if config_value == option:
            return options[config]

def create_nested_dir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        if os.path.isdir(directory):
            pass
        else:
            raise SystemError("os was unable to construct data folder hierarchy")



def read_broken_runs(path):
    with open(path, "rb") as file:
        broken_runs = pickle.load(file)
    return broken_runs



def print_malloc_snapshot(max_nr_stats=10):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    print("top 10")
    for stat in top_stats[:max_nr_stats]:
        print(stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 23)
    parser.add_argument("-r", "--run",
                        default = None)
    parser.add_argument("--debug", action = "store_true")
    
    parser.add_argument("--config", help = "path to config.json file", default = "config.json")
    parser.add_argument("--filename_appendix", default = "")
    
    parser.add_argument("--skip_clean", action = "store_true")
    parser.add_argument("--test", action = "store_true", help = "enables test mode, which only uses one run of data ")
    parser.add_argument("--nr_batches", type=int, default=None, help="only for data, sims are fast enough")
    parser.add_argument("--batch_i", type=int, default=None, help="Only for data, sims are fast enough")
    args = parser.parse_args()


    with open(args.config, "r") as config_json:
        config = json.load(config_json)


    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level = log_level)


    calculate_variable = functions[config["variable"]]

    logger.debug("Initialising detector")
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=args.station,
                            log_level=log_level)
    
    logger.debug("Updating detector time")
    det.update(Time(config["detector_time"]))
    

    broken_runs = read_broken_runs(config['broken_runs_dir'] + f"/station{args.station}.pickle")
    broken_runs_list = [int(run) for run in broken_runs.keys()]

    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir


    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*")
        run_files = glob.glob(f"{data_dir}/station{args.station}/run**/*", recursive=True)
        if np.any([run_file.endswith(".root") for run_file in run_files]):
            root_dirs = [root_dir for root_dir in root_dirs if not int(os.path.basename(root_dir).split("run")[-1]) in broken_runs_list]
            root_dirs = sorted(root_dirs)
            if args.nr_batches is not None:
                root_dirs = np.array(root_dirs)
                root_dirs = np.array_split(root_dirs, args.nr_batches)
                root_dirs = root_dirs[args.batch_i]
            channels_to_include = list(np.arange(24))
            config["channels_to_include"] = channels_to_include
        elif np.any([run_file.endswith(".nur") for run_file in run_files]):
            config["simulation"] = True
            sim_config_path = glob.glob(f"{data_dir}/config*")[0]
            with open(sim_config_path, "r") as sim_config_file:
                sim_config = json.load(sim_config_file)
            config.update(sim_config)
            
            noise_sources = config["noise_sources"]
            root_dirs_list = []
            for noise_source in noise_sources:
                root_dirs_tmp = [glob.glob(f"{root_dir}/events_{noise_source}_batch*")[0] for root_dir in root_dirs]
                root_dirs_list.append(root_dirs_tmp)
            if config["include_sum"]:
                root_dirs_tmp = [glob.glob(f"{root_dir}/events_batch*")[0] for root_dir in root_dirs]
                root_dirs_list.append(root_dirs_tmp)
        else:
            raise TypeError("Data extension not recognized")



    if args.test:
        root_dirs = root_dirs[300:305]
        print(root_dirs)

    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]



    if np.any([run_file.endswith(".root") for run_file in run_files]):
        calibration = config["calibration"][str(args.station)]
        mattak_kw = config["mattak_kw"]
        # note if no runtable provided, runtable is queried from the database
        rnog_reader = dataProviderRNOG()
        rnog_reader.begin(root_dirs,
                          reader_kwargs = dict(
                          selectors=selectors,
                          read_calibrated_data=calibration == "full",
                          apply_baseline_correction="approximate",
                          convert_to_voltage=calibration == "linear",
                          select_runs=True,
                          run_types=["physics"],
                          run_time_range=run_time_range,
                          max_trigger_rate=config["max_trigger_rate"] * units.Hz,
                          mattak_kwargs=mattak_kw),
                          det=det)
        rnog_reader = [rnog_reader]
        folder_appendix = [None]
    elif np.any([run_file.endswith(".nur") for run_file in run_files]):
        rnog_reader = []
        for root_dirs in root_dirs_list:
            reader = eventReader()
            reader.begin(root_dirs)
            rnog_reader.append(reader)
        folder_appendix = noise_sources + ["sum"]


    function = functions[config["variable"]]
    t0 = time.time()
    for i, reader in enumerate(rnog_reader):
        output = parse_data(reader, det, config=config, args=args, logger=logger, calculation_function=function, folder_appendix=folder_appendix[i])
    dt = time.time() - t0

    if np.any([run_file.endswith(".root") for run_file in run_files]):
        logger.debug(f"code took {dt} for {len(reader.reader.get_events_information())} events")

#    output = parse_data(rnog_reader, det, config=config, args=args, logger=logger, calculation_function=populate_spec_amplitude_histogram)
    
# test
#    print(output.shape)
#    print(output[0][400])
#
#    function_kwargs = config["variable_function_kwargs"]["clean"]
#    hist_range= function_kwargs["hist_range"]
#    nr_bins= function_kwargs["nr_bins"]
#    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
#    plt.stairs(output[0, 400], edges=bin_edges) 
#    plt.savefig("test")
#    print_malloc_snapshot()
