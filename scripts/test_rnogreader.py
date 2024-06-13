import logging
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

from NuRadioReco.utilities import units
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

def plot_trace(ax, trace, channel_id, read_calibrated_data = False):
    ax.plot(trace, c = "b")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude (mV)" if read_calibrated_data
                  else "amplitude (ADC)")
    ax.set_title(f"channel {channel_id}")
    return ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "script to test NRO_G mattak data reader")
    parser.add_argument("-d", "--data_dir",
                        default = "/pnfs/iihe/rno-g/data/handcarry22/station24/run1")
    parser.add_argument("-c", "--calibrated",
                        action = "store_true",)
    args = parser.parse_args()

    rnog_reader = readRNOGData(log_level=logging.DEBUG)
    # Choose selectors to filter data, these can be callable or list of callable
    selectors = lambda event_info : event_info.triggerType == "FORCE"
    # These are the default mattak kwargs, these can be changed to test different data reading aspects
    mattak_kw = dict(backend = "auto", read_daq_status = True)
    rnog_reader.begin(args.data_dir,
                      selectors = selectors,
                      read_calibrated_data = args.calibrated,
                      apply_baseline_correction = "approximate",
                      convert_to_voltage = False,
                      select_runs = True,
                      run_types = ["physics"],
                      max_trigger_rate = 2 * units.Hz,
                      mattak_kwargs = mattak_kw)
    
    for i_event, event in enumerate(rnog_reader.run()):
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        # picking out a random event
        # let the whole for loop finish for testing purposes
        if i_event == 3:
            print("plotting event")
            fig, axs = plt.subplots(6, 4, figsize = (20, 12))
            axs = np.ndarray.flatten(axs)
            for i_channel, channel in enumerate(station.iter_channels()):
                channel_id = channel.get_id()
                trace = channel.get_trace()
                plot_trace(axs[i_channel], trace, channel_id, read_calibrated_data = args.calibrated)
            fig.tight_layout()
            fig_path = os.path.abspath(f"{os.path.dirname(__file__)}/../figures")
            fig.savefig(f"{fig_path}/test_rnogreader.png")
            break