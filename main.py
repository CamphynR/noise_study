"""
This file is written to process RNOG data using the NuRadio modules, in particular the readRNOGDataMattak module.
One can choose cleaning modules (either homemade or directly frim NuRadio) and a variable to process and the script will
run over all available data, calculate the variable and store it in a pickle file.

Whether all variables or only the mean over all events is to be stored can be chosen in the argument parser using the --only_mean flag
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
import numpy as np
from sys import getsizeof
from typing import Callable
from numba import jit
from astropy.time import Time

import NuRadioReco
import NuRadioReco.modules
import NuRadioReco.modules.RNO_G
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator

import modules.cwFilter

logging.basicConfig(level = logging.WARNING)

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

def calculate_spec_hist(channel, range, nr_bins):
    sampling_rate = channel.get_sampling_rate()
    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()
    spec = np.abs(spec)

    bins = np.linspace(range[0], range[1], nr_bins)
    bin_idxs = np.digitize(spec, bins)
    return freqs, bin_idxs, sampling_rate

def calculate_trace_hist(channel, range, nr_bins):
    bins = np.linspace(range[0], range[1], nr_bins)
    trace = channel.get_trace()
    hist, edges = np.histogram(trace, bins=bins)
    return hist, edges

def select_config(config_value : str, options : dict) -> np.ndarray:
    for option in options.keys():
        if config_value == option:
            return options[config]

def create_nested_dir(dir):
    try:
        os.makedirs(dir)
    except:
        if os.path.isdir(dir):
            pass
        else:
            raise SystemError("os was unable to construct data folder hierarchy")

def read_broken_runs(path):
    with open(path, "rb") as file:
        broken_runs = pickle.load(file)
    return broken_runs



def parse_variables(reader, detector, config, args,
                    calculate_variable = calculate_trace,):
    clean_data = not args.skip_clean
    # initialise cleaning modules
    cleaning_options = {"channelBandpassFilter" : NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter,
                        "hardwareResponseIncorporator" : NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator,
                        "cwFilter" : modules.cwFilter.cwFilter}
    cleaning_modules = dict((cf, cleaning_options[cf]()) for cf in config["cleaning"].keys())

    for cleaning_key in cleaning_modules.keys():
        cleaning_modules[cleaning_key].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])

    logger.debug("Starting calculation")
    if config["only_mean"]:
        variables_list = initialise_variables_list(calculate_variable)
        squares_list = initialise_variables_list(calculate_variable)
    else:
        variables_list = []
    
    clean = "raw" if args.skip_clean else "clean"
    kwargs = config["variable_function_kwargs"][clean]

    # initialise data folder

    if config["save"]:
        # assumes function name to be calculate_*
        function_name = calculate_variable.__name__.split("_", 1)[1]
#        save_dir = config["save_dir"]
        # first save to /tmp directory native to computer node and afterwards copy to pnfs
        save_dir = "/tmp"
        # defining a savename overwrites the data structure and simply saves everything in a file,
        # useful when running one script for multiple stations
        if config["savename"]:
            filename = f"{save_dir}/variable_lists/{function_name}_lists/{config['savename']}.pickle"
        else:
            date = datetime.datetime.now().strftime("%Y_%m_%d")
            dir = f"{save_dir}/{function_name}/job_{date}"
            if args.test:
                dir += "_test"
            create_nested_dir(dir)

            # fill directory
            config_name = f"{dir}/config.json"
            station_ids = np.array(detector.get_station_ids())
            for station_id in station_ids:
                station_dir = f"{dir}/station{station_id}/{clean}"
                if not os.path.exists(station_dir):
                    os.makedirs(station_dir)

        # copies the config used for future reference
        settings_dict = {**config, **vars(args)}
        if not os.path.isfile(config_name):
            with open(config_name, "w") as f:
                json.dump(settings_dict, f)

    t0 = time.time()

    events_processed = 0
    for event in reader.run():
        if events_processed == 0:
            prev_run_nr = event.get_run_number()
        events_processed += 1
        run_nr = event.get_run_number()
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        logger.debug(f"Trigger is: {station.get_triggers()}")
        logger.debug(f"Run number is {event.get_run_number()}")
        print(f"variables_list size is {getsizeof(variables_list)/1000} kB")
        if prev_run_nr != run_nr:
            if config["save"]:
                logger.debug(f"saving since {run_nr} != {prev_run_nr}")
                filename = f"{dir}/station{station_id}/{clean}/run{prev_run_nr}"
                filename += ".pickle"
                print(f"Saving as {filename}")
                with open(filename, "wb") as f:
                    pickle.dump(dict(time=station.get_station_time(), var=variables_list), f)

                if config["only_mean"]:
                    variables_list = initialise_variables_list(calculate_variable)
                    squares_list = initialise_variables_list(calculate_variable)
                else:
                    variables_list = []

        # there should be a mechanism in the detector code which makes sure
        # not to reload the detector for the same time stamps
        station_time = station.get_station_time()
        det.update(station_time)

        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector, **config["cleaning"][cleaning_key]["run_kwargs"])
       
        var_channels_per_event = []
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if config["only_mean"]:
                variables_list[channel_id, :] += calculate_variable(channel, **kwargs)
                squares_list[channel_id, :] += calculate_variable(channel, **kwargs)**2
            else:
                var_channels_per_event.append(calculate_variable(channel, **kwargs))

        variables_list.append(var_channels_per_event)
        
        prev_run_nr = event.get_run_number()



    if config["save"]:
        logger.debug(f"saving since {run_nr} != {prev_run_nr}")
        filename = f"{dir}/station{station_id}/{clean}/run{prev_run_nr}"
        filename += ".pickle"
        print(f"Saving as {filename}")
        with open(filename, "wb") as f:
            pickle.dump(dict(time=station.get_station_time(), var=variables_list), f)
        
        src_dir = "/tmp/"
        dest_dir = config["save_dir"]
        subprocess.call(["rsync", "-vuar", src_dir, dest_dir])
        

    if config["only_mean"]:
        print(f"total events that passed filter {events_processed}")
        variables_list = variables_list/events_processed
        var_list = squares_list/events_processed - variables_list**2
    

    dt = time.time() - t0
    logger.debug(f"Main calculation loop takes {dt}")

    return variables_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 24)
    parser.add_argument("-r", "--run",
                        default = None)
    parser.add_argument("--debug", action = "store_true")
    
    parser.add_argument("--config", help = "path to config.json file", default = "config.json")
    
    parser.add_argument("--skip_clean", action = "store_true")
    parser.add_argument("--test", action = "store_true", help = "enables test mode, which only uses one run of data ")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level = log_level)


    functions = dict(rms = calculate_rms,
                     trace = calculate_trace,
                     spec = calculate_spec,
                     trace_hist = calculate_trace_hist,
                     spec_hist = calculate_spec_hist)
    calculate_variable = functions[config["variable"]]

    logger.debug("Initialising detector")
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=args.station,
                            log_level=log_level)
    
    logger.debug("Updating detector time")
    det.update(Time(config["detector_time"]))
    
    # note if no runtable provided, runtable is queried from the database
    rnog_reader = readRNOGData(log_level=log_level)

    broken_runs = read_broken_runs(config['broken_runs_dir'] + f"/station{args.station}.pickle")
    broken_runs_list = [int(run) for run in broken_runs.keys()]

    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*[!run363]") # run 363 is broken (100 waveforms with 200 event infos)
        root_dirs = [root_dir for root_dir in root_dirs if not int(os.path.basename(root_dir).split("run")[-1]) in broken_runs_list]

    if args.test:
        root_dirs = root_dirs[:3]

    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]

    calibration = config["calibration"][str(args.station)]

    mattak_kw = dict(backend="pyroot", read_daq_status=False, read_run_info=False)
    rnog_reader.begin(root_dirs,
                      selectors=selectors,
                      read_calibrated_data=calibration == "full",
                      apply_baseline_correction="approximate",
                      convert_to_voltage=calibration == "linear",
                      select_runs=True,
                      run_types=["physics"],
                      run_time_range=run_time_range,
                      max_trigger_rate=2 * units.Hz,
                      mattak_kwargs=mattak_kw)


    rms = parse_variables(rnog_reader, det, config=config, args=args,
                          calculate_variable=calculate_variable)
