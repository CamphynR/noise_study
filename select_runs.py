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


def read_broken_runs(path):
    with open(path, "rb") as file:
        broken_runs = pickle.load(file)
    return broken_runs


if __name__ == "__main__":
    tracemalloc.start()

    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 23)
    
    parser.add_argument("--config", help = "path to config.json file", default = "config.json")
    args = parser.parse_args()


    with open(args.config, "r") as config_json:
        config = json.load(config_json)


    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG
    logging.basicConfig(level = log_level)


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


    root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*")
    root_dirs = [root_dir for root_dir in root_dirs if not int(os.path.basename(root_dir).split("run")[-1]) in broken_runs_list]
    root_dirs = sorted(root_dirs)
    channels_to_include = list(np.arange(24))
    config["channels_to_include"] = channels_to_include


    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]


    calibration = config["calibration"][str(args.station)]
    mattak_kw = dict(backend="pyroot", read_daq_status=False, read_run_info=False)
    # note if no runtable provided, runtable is queried from the database
    rnog_reader = dataProviderRNOG()
    print("beginning reader")
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

    runs = rnog_reader.reader.get_events_information("run")
    runs = np.unique(runs)
    print(runs)

    json_path = f"selected_runs_station{args.station}.json"
    with open(json_path, "w") as json_file:
        json.dump(list(runs), json_file)
