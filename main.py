"""
This file is written to process RNOG data using the NuRadio modules, in particular the readRNOGDataMattak module.
One can choose cleaning modules (either homemade or directly frim NuRadio) and a variable to process and the script will
run over all available data, calculate the variable and store it in a pickle file.

Whether all variables or only the mean over all events is to be stored can be chosen in the argument parser using the --only_mean flag
"""


import os
import time
import logging

import NuRadioReco.modules
import NuRadioReco.modules.RNO_G
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
logging.basicConfig(level = logging.WARNING)
import glob
import json
import numpy as np
import pickle
import argparse
import datetime
from typing import Callable
import matplotlib.pyplot as plt
from numba import jit

from astropy.time import Time
import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator

import modules.cwFilter

def get_output_shape(function : Callable[[BaseTrace], np.ndarray]):
    """
    Function to obtain the output shape of a function that applies to a NuRadio BaseTrace object
    """
    dummy_channel = BaseTrace()
    dummy_trace = np.ones(2048)
    dummy_fs = 3.2e9 * units.Hz
    dummy_channel.set_trace(dummy_trace, sampling_rate = dummy_fs)
    dummy_output = function(dummy_channel)
    #check to get shape (1) of output is a single entity e.g. an int or float
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
        std_list = initialise_variables_list(calculate_variable)
    else:
        variables_list = [[] for c in range(24)]

    t0 = time.time()

    events_processed = 0
    for event in reader.run():
        events_processed += 1
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        print(station.get_triggers())
        print(event.get_run_number())
        # station_time = station.get_station_time()
        # if events_processed == 1:
        #     start_time = station_time
        # dt = station_time - start_time
        # if dt.to_value("jd", "long") > 365:
        #     det.update(station_time)
        
        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector, **config["cleaning"][cleaning_key]["run_kwargs"] )
            
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if config["only_mean"]:
                variables_list[channel_id, :] += calculate_variable(channel)
                std_list[channel_id, :] += calculate_variable(channel)**2
            else:
                variables_list[channel_id].append(calculate_variable(channel))

    if config["only_mean"]:
        print(f"total events that passed filter {events_processed}")
        variables_list = variables_list/events_processed
        # std_list = 
    

    dt = time.time() - t0
    logger.debug(f"Main calculation loop takes {dt}")

    if config["save"]:
        # assumes function name to be calculate_*
        function_name = calculate_variable.__name__.split("_")[1]
        save_dir = config["save_dir"]
        # defining a savename overwrites the data structure and simplu saves everything in a file,
        # useful when running one script for multiple stations
        if config["savename"]:
            filename = f"{save_dir}/variable_lists/{function_name}_lists/{config['savename']}.pickle"
        else:
            date = datetime.datetime.now().strftime("%Y_%m_%d")
            dir = f"{save_dir}/{function_name}/run_{date}"
            if args.test:
                dir += "_test"

            create_nested_dir(dir)

            config_name = f"{dir}/config_s{args.station}.json"
            # assumes only one station is run at a time
            station_ids = np.array(detector.get_station_ids())
            filename = f"{dir}/station{station_ids[0]}"
            if not clean_data:
                filename += "_no_filters"
            filename += ".pickle"
        print(f"Saving as {filename}")
        with open(filename, "wb") as f:
            pickle.dump(variables_list, f)
        # copies the config used for future reference
        settings_dict = {**config, **vars(args)}
        with open(config_name, "w") as f:
            json.dump(settings_dict, f)

    return np.array(variables_list)



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


    functions = dict(rms = calculate_rms, trace = calculate_trace, spec = calculate_spec)
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
    print(broken_runs_list)

    if args.data_dir == None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*[!run363]") # run 363 is broken (100 waveforms with 200 event infos)
        root_dirs = [root_dir for root_dir in root_dirs if not int(os.path.basename(root_dir).split("run")[-1]) in broken_runs_list]

    if args.test:
        root_dirs = root_dirs[:10]

    print(root_dirs)
    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]
    
    mattak_kw = dict(backend = "pyroot", read_daq_status = False)
    rnog_reader.begin(root_dirs,
                      selectors=selectors,
                      read_calibrated_data=config["calibration"] == "full",
                      apply_baseline_correction="approximate",
                      convert_to_voltage=config["calibration"] == "linear",
                      select_runs=True,
                      run_types=["physics"],
                      run_time_range=run_time_range,
                      max_trigger_rate=2 * units.Hz,
                      mattak_kwargs=mattak_kw)

    # cleaning parameters
    passband = [200 * units.MHz, 600 * units.MHz]      

    rms = parse_variables(rnog_reader, det, config=config, args=args,
                          calculate_variable = calculate_variable)
