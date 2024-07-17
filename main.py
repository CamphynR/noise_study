"""
This file is written to process RNOG data using the NuRadio modules, in particular the readRNOGDataMattak module.
One can choose cleaning modules (either homemade or directly frim NuRadio) and a variable to process and the script will
run over all available data, calculate the variable and store it in a pickle file.

Whether all variables or only the mean over all events is to be stored can be chosen in the argument parser using the --only_mean flag
"""


import datetime
import os
import time
import logging
import glob
import numpy as np
import pickle
import argparse
from typing import Callable

import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.framework.base_trace import BaseTrace

import modules.filter_cw
import modules.filter_detector

def get_output_shape(function : Callable[[BaseTrace], ...]):
    """
    Function to obtain the output shape of a function that applies to a NuRadio BaseTrace object
    """
    dummy_channel = BaseTrace()
    dummy_trace = np.ones(2048)
    dummy_fs = 3.2e9 * units.Hz
    dummy_channel.set_trace(dummy_trace, sampling_rate = dummy_fs)
    dummy_output = function(dummy_channel)
    return np.array(dummy_output).shape

def initialise_variables_list(function, nr_of_stations, nr_of_channels = 24):
    output_shape = get_output_shape(function)
    list_shape = tuple([nr_of_stations, nr_of_channels] + list(output_shape))
    variables_list = np.zeros(list_shape)
    return variables_list

def calculate_rms(channel):
    trace = channel.get_trace()
    rms = np.sqrt(np.mean(trace**2))
    return rms

def calculate_trace(channel):
    return channel.get_trace()

def calculate_spec(channel):
    spec = channel.get_frequency_spectrum()
    return np.abs(spec)

def parse_variables(reader, detector, passband, save = False, clean_data = True,
                    calculate_variable = calculate_trace, only_mean = True,
                    test = False):
 
    # initialise cleaning modules
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    detectorFilter = modules.filter_detector.detectorFilter()
    detectorFilter.begin(detector)
    cwFilter = modules.filter_cw.cwFilter()
    cwFilter.begin()
    
    logger.debug("Starting calculation")
    station_ids = np.array(detector.get_station_ids())
    if only_mean:
        variables_list = initialise_variables_list(calculate_variable, len(station_ids))
    else:
        variables_list = [[[] for c in range(24)] for s in station_ids]

    t0 = time.time()

    events_processed = 0
    for event in reader.run():
        events_processed += 1
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)

        if clean_data:
            channelBandPassFilter.run(event, station, detector, passband = passband)
            detectorFilter.run(event, station)
            cwFilter.run(event, station)
    
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if only_mean:
                variables_list[station_id == station_ids, channel_id, :] += calculate_variable(channel)
            else:
                idx = np.where(station_id == station_ids)[0][0] # needed because here variables_list is list type
                variables_list[idx][channel_id].append(calculate_variable(channel))

    if only_mean:
        print(f"total events that passed filter {events_processed}")
        variables_list = variables_list/events_processed

    dt = time.time() - t0
    logger.debug(f"Main calculation loop takes {dt}")

    if save:  
        logger.debug("Saving rms to file")

        # assumes function name to be calculate_*
        function_name = calculate_variable.__name__[10:]
        station_names = "-".join([str(s) for s in station_ids])
        appendix = "clean" if clean_data else "unclean"
        if only_mean:
            appendix += "_mean"
        if test:
            appendix += "_test"    
        filename = f"variable_lists/{function_name}_lists/{function_name}_s{station_names}_{appendix}.pickle"
        print(f"Saving as {filename}")
        with open(filename, "wb") as f:
            pickle.dump(variables_list, f)

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
                        type = int,
                        default = None)
    parser.add_argument("-v", "--variable",
                        required = True,
                        choices = ["rms", "trace", "spec"])      
    parser.add_argument("--only_mean", action = "store_true")          
    parser.add_argument("-c", "--calibration",
                        choices = ["linear", "full"],
                        default = "linear")
    parser.add_argument("--save", action = "store_true",
                        help = "save to pickle")
    parser.add_argument("--skip_clean", action = "store_true")
    parser.add_argument("--test", action = "store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.DEBUG)

    functions = dict(rms = calculate_rms, trace = calculate_trace, spec = calculate_spec)
    calculate_variable = functions[args.variable]

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(2022, 7, 15))

    rnog_reader = readRNOGData(log_level = logging.DEBUG) #note if no runtable provided, runtable is queried from the database

    if args.data_dir == None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    elif args.test:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run1/")
    else:    
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*[!run363]/") # run 363 is broken (100 waveforms with 200 event infos)

   
    
    selectors = lambda event_info : event_info.triggerType == "FORCE"
    mattak_kw = dict(backend = "uproot", read_daq_status = False)
    rnog_reader.begin(root_dirs,    
                      selectors = selectors,
                      read_calibrated_data = args.calibration == "full",
                      apply_baseline_correction="approximate",
                      convert_to_voltage = args.calibration == "linear",
                      select_runs = True,
                      run_types = ["physics"],
                      max_trigger_rate = 2 * units.Hz,
                      mattak_kwargs = mattak_kw)

    # cleaning parameters
    passband = [200 * units.MHz, 600 * units.MHz]               
    
    rms = parse_variables(rnog_reader, det, passband, save = args.save,
                          clean_data = not args.skip_clean,
                          calculate_variable = calculate_variable,
                          only_mean = args.only_mean,
                          test = args.test)