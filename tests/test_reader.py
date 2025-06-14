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
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
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
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    log_level = logging.WARNING
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
    mattak_kw = dict(backend="uproot", read_daq_status=False, read_run_info=False, cache_calibration=False)

    # note if no runtable provided, runtable is queried from the database
    rnog_reader = dataProviderRNOG()
    logger.debug("beginning reader")
    selectors = [lambda eventInfo : eventInfo.triggerType == "FORCE"]
    rnog_reader.begin(root_dirs,
                        reader_kwargs = dict(
                        read_calibrated_data=calibration == "full",
                        apply_baseline_correction="approximate",
                        convert_to_voltage=calibration == "linear",
                        selectors = selectors,
                        mattak_kwargs=mattak_kw,
                        max_trigger_rate=2 * units.Hz
                        ),
                        det=det,)

    sine_subtraction = channelSinewaveSubtraction()
    sine_subtraction.begin(save_filtered_freqs=True)
    sine_sub_kw = {
            "algorithm" : "simple",
            "peak_prominence" : 4}

    plt.style.use("gaudi")
    channel_id = 13
    for event in rnog_reader.run():
        print(event.get_id())
#        if event.get_id() == 132:
#        if event.get_id() == 12:
        if event.get_id() == 1288:
            station = event.get_station()
            station = event.get_station()
            channel = station.get_channel(channel_id)
            freq = channel.get_frequencies()
            ft = channel.get_frequency_spectrum()
            plt.plot(freq, np.abs(ft), label="before cw subtraction")
            sine_subtraction.run(event, station, det, **sine_sub_kw)
            station = event.get_station()
            channel = station.get_channel(channel_id)
            freq = channel.get_frequencies()
            ft = channel.get_frequency_spectrum()
            plt.plot(freq, np.abs(ft), label="after cw subtraction", alpha = 0.6)
            plt.legend()
            plt.xlim(0, 1.)
            plt.savefig("figures/tests/test_ft")
            plt.close()

            times = channel.get_times()
            trace = channel.get_trace()
            plt.plot(times, trace)
            plt.savefig("figures/tests/test_trace")
            break
    print(sine_subtraction.removed_freqs)
    rnog_reader.end()
    del rnog_reader
