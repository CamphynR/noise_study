from astropy.time import Time
import argparse
import datetime
import glob
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from NuRadioReco.detector import detector
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.modules.io.RNO_G import readRNOGDataMattak
from NuRadioReco.utilities import units
#from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData


logging.basicConfig(level = logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 23)
    parser.add_argument("-r", "--run",
                        default = None,
                        nargs = "+")
    parser.add_argument("--debug", action = "store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level = log_level)


    logger.debug("Initialising detector")
    det = detector.Detector(source="rnog_mongo",
                            always_query_entire_description=False,
                            database_connection="RNOG_public",
                            select_stations=args.station,
                            log_level=log_level)
    logger.debug("Updating detector time")
    det.update(Time("2022-08-01"))


    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        if len(args.run) == 1:
            root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run[0]}/")
        else:
            root_dirs = []
            for run_nr in args.run:
                root_dirs.append(glob.glob(f"{data_dir}/station{args.station}/run{run_nr}/")[0])
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*")
        channels_to_include = list(np.arange(24))



    calibration = "linear"
    mattak_kw = dict(backend="pyroot", read_daq_status=False, read_run_info=False)

    # note if no runtable provided, runtable is queried from the database
    rnog_reader = dataProviderRNOG()
    logger.debug("beginning reader")
    rnog_reader.begin(root_dirs,
                        reader_kwargs = dict(
                        read_calibrated_data=calibration == "full",
                        apply_baseline_correction="approximate",
                        convert_to_voltage=calibration == "linear",
                        mattak_kwargs=mattak_kw),
                        det=det)

    for event in rnog_reader.run():
        print(event.get_id())