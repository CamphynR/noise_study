import datetime
import os
import logging
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_axes_aligner import align

import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter

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
                        default = 1)
    parser.add_argument("-e", "--event",
                        type = int, default = 0)
    parser.add_argument("-c", "--channel",
                        type = int, default = 0)
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

    root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")

    print(root_dirs)
    
    mattak_kw = dict(backend = "uproot", read_daq_status = False)
    rnog_reader.begin(root_dirs,
                      read_calibrated_data = False,
                      apply_baseline_correction="approximate",
                      convert_to_voltage = True,
                      mattak_kwargs = mattak_kw)
    
    event = rnog_reader.get_event_by_index(args.event)

    channelBandPassFilter = channelBandPassFilter()
    passband = [200 * units.MHz, 600 * units.MHz]

    station = event.get_station()
    station_id = station.get_id()

    channelBandPassFilter.run(event, station, det, passband = passband)

    channel = station.get_channel(args.channel)     
    channel_id = channel.get_id()
    trace = channel.get_trace()


    detector_response = det.get_signal_chain_response(station_id, channel_id)
    fs = 3.2e9

    freq = channel.get_frequencies()

    fig, ax = plt.subplots(2, 2, figsize = (16, 12))

    times = (np.arange(2048) / fs) * units.s
    ax[0][0].plot(times/units.ns, trace, label = "before")
    ax[0][0].set_xlabel("time / ns", size = "x-large")
    ax[0][0].set_ylabel("amplitude / V", size = "x-large")
    ax[0][0].legend(loc = "best")


    
    freq = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()
    ax[1][0].plot(freq, np.abs(spec), label = "before")
    ax[1][0].set_xlabel("freq / GHz")
    ax[1][0].set_ylabel("amlitude / V/GHz")
    ax[1][0].set_xlim(0.1, 0.7)
    ax[1][0].legend(loc = "best")


    test_channel = channel / detector_response
    test_freq = test_channel.get_frequencies()
    test_spec = test_channel.get_frequency_spectrum()
    test_trace = test_channel.get_trace()
    ax[0][1].plot(times / units.ns, test_trace, label = "after removing detector", c = "orange")
    ax[0][1].set_xlabel("time / ns", size = "x-large")
    ax[0][1].set_ylabel("amplitude / V", size = "x-large")
    ax[0][1].legend(loc = "best")

    ax[1][1].plot(test_freq, np.abs(test_spec), label = "after", c = "orange")
    ax[1][1].set_xlabel("freq / GHz", size = "x-large")
    ax[1][1].set_ylabel("amplitude / V/GHz", size = "x-large")
    ax[1][1].set_xlim(0.1, 0.7)
    ax[1][1].legend(loc = "best")

    figdir = os.path.abspath("{__file__}/../figures/test_det_deconv")   
    fig.savefig(figdir, bbox_inches = "tight")

    fig_sec, ax_sec = plt.subplots()
    detector_response.plot(ax1 = ax_sec, in_dB = False)
    figdir_sec = os.path.abspath("{__file__}/../figures/test_det_deconv_response")
    fig_sec.savefig(figdir_sec, bbox_inches = "tight")

    fig_ter, ax_ter = plt.subplots(1, 2, figsize = (18, 8))
    ax_ter_after = [ax_ter[0].twinx(), ax_ter[1].twinx()]

    pl0, = ax_ter[0].plot(times / units.ns, trace, label = "before removing detector")
    ax_ter[0].set_xlabel("time / ns", size = "x-large")
    ax_ter[0].set_ylabel("amplitude / V", size = "x-large")
  
    pl0after, = ax_ter_after[0].plot(times / units.ns, test_trace, label = "after", ls = "dashed", alpha = 0.5, c = "r")
    ax_ter_after[0].tick_params(axis = "y", labelcolor = "red")    
    labels0 = [pl0.get_label(), pl0after.get_label()]
    ax_ter[0].legend([pl0, pl0after], labels0 ,loc = 1)

    pl1, = ax_ter[1].plot(freq, np.abs(spec), label = "before")
    ax_ter[1].set_xlabel("freq / GHz", size = "x-large")
    ax_ter[1].set_ylabel("amplitude / V/GHz", size = "x-large")
    ax_ter[1].set_xlim(0.15, 0.65)

    pl1after, = ax_ter_after[1].plot(test_freq, np.abs(test_spec), label = "after", ls = "dashed", alpha = 0.5, c = "r")
    ax_ter_after[1].tick_params(axis = "y", labelcolor = "red")
    labels1 = [pl1.get_label(), pl1after.get_label()]
    ax_ter[1].legend([pl1, pl1after], labels1 ,loc = 1)

    align.yaxes(ax_ter[0], 0., ax_ter_after[0], 0.)

    figdir_ter = os.path.abspath("{__file__}/../figures/test_det_deconv_overlay")
    fig_ter.savefig(figdir_ter, bbox_inches = "tight")