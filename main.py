mport datetime
import os
import logging
import glob
import numpy as np
import pickle
import argparse

import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

from modules.filter_cw import cwFilter


def read_rms(reader, station_idx, detector, passband, save = False, clean_data = True):
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    cwFilter = cwFilter()
    cwFilter.begin()

    logger.debug("Starting rms calculation")
    rms_list = [[] for i in range(24)]
    for i_event, event in enumerate(reader.run()):
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        if clean_data:
            channelBandPassFilter.run(event, station, detector, passband = passband)
            cwFilter.run()

        for channel in station.iter_channels():
            channel_id = channel.get_id()
            trace = channel.get_trace()
            rms = np.sqrt(np.mean(trace**2))
            rms_list[channel_id].append(rms)

    logger.debug("Saving rms to file")
    
    if save:
        appendix = "clean" if clean_data else "unclean"
        filename = f"rms_lists/rms_s{station_idx}_{appendix}.pickle"
        with open(filename, "wb") as f:
            pickle.dump(rms_list, f)

    return np.array(rms_list)

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
    parser.add_argument("-c", "--calibration",
                        choices = ["linear", "full"],
                        default = "linear")
    parser.add_argument("--save", action = "store_true",
                        help = "save to pickle")
    parser.add_argument("--skip_clean", action = "store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.DEBUG)


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

    if args.run is None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*[!run363]/") # run 363 is broken (100 waveforms with 200 event infos)
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/") # run 363 is broken (100 waveforms with 200 event infos)

    print(root_dirs)

    # cleaning parameters
    passband = [200 * units.MHz, 1000 * units.MHz]
    
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
    
    rms = read_rms(rnog_reader, args.station, det, passband, args.save,
                   clean_data = not args.skip_clean)