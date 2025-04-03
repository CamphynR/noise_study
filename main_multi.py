"""
This file is written to process RNOG data using the NuRadio modules, in particular the readRNOGDataMattak module.
One can choose cleaning modules (either homemade or directly frim NuRadio) and a variable to process and the script will
run over all available data, calculate the variable and store it in a pickle file.

Whether all variables or only the mean over all events is to be stored can be chosen in the argument parser using the --only_mean flag

The whole script uses NuRadio base units (freq in GHz, time in ns)
"""

from astropy.time import Time
import argparse
import copy
import datetime
import glob
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import time
import tracemalloc
from typing import Callable
import subprocess

import NuRadioReco
from NuRadioReco.detector import detector
from NuRadioReco.framework.base_trace import BaseTrace
import NuRadioReco.modules
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.io.eventReader import eventReader
import NuRadioReco.modules.RNO_G
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units
#from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

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


def select_config(config_value : str, options : dict) -> np.ndarray:
    for option in options.keys():
        if config_value == option:
            return options[config]


def open_select_runs_list(select_runs_path):
    with open(select_runs_path, "r") as select_runs_file:
        select_runs_list = json.load(select_runs_file)
    return select_runs_list

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
    tracemalloc.start()

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
    parser.add_argument("--select_runs_path", help = "path to select_runs json file", default=None)
    parser.add_argument("--filename_appendix", default = "", type=str)
    
    parser.add_argument("--skip_clean", action = "store_true")
    parser.add_argument("--test", action = "store_true", help = "enables test mode, which only uses one run of data ")
    parser.add_argument("--nr_batches", type=int, default=None, help="only for data, sims are fast enough")
    parser.add_argument("--batch_i", type=int, default=None, help="Only for data, sims are fast enough")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level = log_level)

    with open(args.config, "r") as config_json:
        config = json.load(config_json)



    calculate_variable = functions[config["variable"]]



    logger.debug("Initialising detector")
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=args.station,
                            log_level=log_level)
    logger.debug("Updating detector time")
    det.update(Time(config["detector_time"]))
    





    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        if args.select_runs_path is not None:
            select_runs_list = open_select_runs_list(args.select_runs_path)
            root_dirs = [glob.glob(f"{data_dir}/station{args.station}/run{run}")[0] for run in select_runs_list]
        else:
            root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*")

        run_files = glob.glob(f"{data_dir}/station{args.station}/run**/*", recursive=True)
        if not len(run_files):
            run_files = glob.glob(f"{data_dir}/station{args.station}/clean/*")

        is_root = np.any([run_file.endswith(".root") for run_file in run_files])
        is_nur = np.any([run_file.endswith(".nur") for run_file in run_files])
        if is_root:
            broken_runs = read_broken_runs(config['broken_runs_dir'] + f"/station{args.station}.pickle")
            broken_runs_list = [int(run) for run in broken_runs.keys()]
            root_dirs = [root_dir for root_dir in root_dirs if not int(os.path.basename(root_dir).split("run")[-1]) in broken_runs_list]
            root_dirs = sorted(root_dirs, key=lambda root_dir : int(os.path.basename(root_dir)[3:]))
            if args.nr_batches is not None:
                root_dirs = np.array(root_dirs)
                root_dirs = np.array_split(root_dirs, args.nr_batches)
            if args.test:
                root_dirs = root_dirs[:3]
            channels_to_include = list(np.arange(24))
            config["channels_to_include"] = channels_to_include
            config["simulation"] = False
        elif is_nur:
            sim_config_path = glob.glob(f"{data_dir}/config*")[0]
            sim_config_path = glob.glob(f"{data_dir}/config*")[0]
            with open(sim_config_path, "r") as sim_config_file:
                sim_config = json.load(sim_config_file)
            config["simulation"] = sim_config.get("simulation", True)
            if config["simulation"]:
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
                if args.nr_batches is not None:
                    root_dirs = np.array(run_files)
                    print(root_dirs)
                    root_dirs = np.array_split(root_dirs, args.nr_batches)
                if args.test:
                    root_dirs = root_dirs[:3]
                channels_to_include = list(np.arange(24))
                config["channels_to_include"] = channels_to_include
                
        else:
            raise TypeError("Data extension not recognized")




    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]

    calibration = config["calibration"][str(args.station)]
    mattak_kw = config["mattak_kw"]

    def batch_process(batch_i):
        print(root_dirs)
        print(batch_i)
        root_dirs_batch = root_dirs[batch_i]
        print(root_dirs_batch)
        # note if no runtable provided, runtable is queried from the database
        rnog_reader = dataProviderRNOG()
        logger.debug("beginning reader")
        try:
            rnog_reader.begin(root_dirs_batch,
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
            args_temp = copy.copy(args)
            args_temp.batch_i = batch_i
            function = functions[config["variable"]]
            output = parse_data(rnog_reader, det, config=config, args=args_temp, logger=logger, calculation_function=function, folder_appendix=None)
            rnog_reader.end()
            del rnog_reader
        except FileNotFoundError:
            del rnog_reader
            return
    
    def batch_process_nur(batch_i):
        root_dirs_batch = root_dirs[batch_i]
        root_dirs_batch = list(root_dirs_batch)
        print(root_dirs_batch)
        # note if no runtable provided, runtable is queried from the database
        rnog_reader = eventReader()
        rnog_reader.begin(root_dirs_batch)
        args_temp = copy.copy(args)
        args_temp.batch_i = batch_i
        function = functions[config["variable"]]
        output = parse_data(rnog_reader, det, config=config, args=args_temp, logger=logger, calculation_function=function, folder_appendix=None)
        rnog_reader.end()
        del rnog_reader
        return

    if args.batch_i is not None:
        print(args.batch_i)
        batch_process(args.batch_i)
    else:
        if is_nur:
            if config["simulation"] == False:
                with multiprocessing.Pool() as p:
                    p.map(batch_process_nur, range(args.nr_batches))
        else:
            with multiprocessing.Pool() as p:
                p.map(batch_process, range(args.nr_batches))
