import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime

import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

def plot_trace(reader, detector, passband : list, clean : bool = False, event_idx : int = 1,
               matplotlib_kw = dict(), fig_dir = "/user/rcamphyn/noise_study/figures"):
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

    # run yields the first event that passes a filter, hence we can break after the first iteration
    for event in reader.run():
        event_id = event.get_id()
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        if clean:
            channelBandPassFilter.run(event, station, detector, passband = passband)

        fig, axs = plt.subplots(4, 6, figsize = (16, 12))
        axs = np.ndarray.flatten(axs) 
        for channel_idx, channel in enumerate(station.iter_channels()):
            channel_id = channel.get_id()
            trace = channel.get_trace() / units.mV
            time = np.arange(2048) / (3.2 * units.GHz) 
            axs[channel_idx].plot(time, trace)
            axs[channel_idx].set_xlabel("time / ns", size = "large")
            axs[channel_idx].set_ylabel("voltage / mV", size = "large")
            axs[channel_idx].set_title(f"channel {channel_idx}")
            print(np.sqrt(np.mean(np.power(trace, 2))))
        
        clean_string = "clean" if clean else "unclean"
        fig.suptitle(f"Time traces for station {station}, event {event_id} ({clean_string})")
        fig.tight_layout()
        fig_file = f"{fig_dir}/trace_s{station_id}_event{event_id}_{clean_string}"
        fig.savefig(fig_file)
        break

def plot_clean_comparison(reader, detector, passband : list, clean : bool = False, event_idx : int = 1,
               matplotlib_kw = dict(), fig_dir = "/user/rcamphyn/noise_study/figures"):
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

    # run yields the first event that passes a filter, hence we can break after the first iteration
    for event in reader.run():
        event_id = event.get_id()
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        fig, axs = plt.subplots(4, 6, figsize = (16, 12))
        axs = np.ndarray.flatten(axs) 

        for channel_idx, channel in enumerate(station.iter_channels()):
            channel_id = channel.get_id()
            trace = channel.get_trace() * 1e3 * units.mV
            time = np.arange(2048) / 3.2 * units.ns 
            axs[channel_idx].plot(time, trace, label = "before clean")
            axs[channel_idx].set_xlabel("time / ns", size = "large")
            axs[channel_idx].set_ylabel("voltage / mV", size = "large")
            axs[channel_idx].set_title(f"channel {channel_idx}")
        
        channelBandPassFilter.run(event, station, detector, passband = passband)

        for channel_idx, channel in enumerate(station.iter_channels()):
            channel_id = channel.get_id()
            trace = channel.get_trace() * 1e-3 * units.mV
            time = np.arange(2048) / 3.2 * units.ns 
            axs[channel_idx].plot(time, trace, label = "after clean", linestyle = "dotted", c = "red", alpha = 0.5)
            axs[channel_idx].set_xlabel("time / ns", size = "large")
            axs[channel_idx].set_ylabel("voltage / mV", size = "large")
            axs[channel_idx].set_title(f"channel {channel_idx}")
        axs[0].legend(loc = "upper right")
        
        clean_string = "clean" if clean else "unclean"
        fig.suptitle(f"Time traces for station {station}, event {event_id} ({clean_string})")
        fig.tight_layout()
        fig_file = f"{fig_dir}/clean_comparison_s{station_id}_event{event_id}"
        fig.savefig(fig_file, dpi = 200)
        break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s',
                                     usage = "independent script to plot time traces and investigate effect of \
                                              different filters or responses (e.g. a bandpassfilter)")
    parser.add_argument("-d", "--data_dir")
    parser.add_argument("-s", "--station",
                        type = int, default = 24)
    parser.add_argument("-c", "--channel")
    parser.add_argument("--clean", action = "store_true")
    parser.add_argument("--calibration", choices = ["full", "linear"],
                        default = "linear")
    parser.add_argument("--compare_cleaning", action = "store_true")
    args = parser.parse_args()

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(2022, 8, 1))

    rnog_reader = readRNOGData(log_level = logging.DEBUG)
    
    # reader options
    # --------------
    selectors = lambda event_info : event_info.triggerType == "FORCE"
    mattak_kw = dict(backend = "uproot")

    rnog_reader.begin(args.data_dir,    
                      selectors = selectors,
                      read_calibrated_data = args.calibration == "full",
                      apply_baseline_correction="approximate",
                      convert_to_voltage = args.calibration == "linear",
                      select_runs = True,
                      run_types = ["physics"],
                      max_trigger_rate = 2 * units.Hz,
                      mattak_kwargs = mattak_kw)

    # cleaning components
    # -------------------
    passband = [200 * units.MHz, 1000 * units.MHz]

    if args.compare_cleaning:
        plot_clean_comparison(rnog_reader, det, passband, clean = args.clean)
    else:
        plot_trace(rnog_reader, det, passband, clean = args.clean)

