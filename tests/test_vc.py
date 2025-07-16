import argparse
import glob
import json
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import mattak.Dataset
import NuRadioReco.utilities.units as units
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 24)
    parser.add_argument("-r", "--run",
                        default = 1920)
    
    parser.add_argument("--config", help = "path to config.json file", default = "config.json")
    
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    rnog_reader = readRNOGData(log_level=logging.DEBUG)

    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    print(f"{data_dir}/station{args.station}")

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        # run 363 is broken (100 waveforms with 200 event infos)
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*[!run363]")[:10]


    print(root_dirs)
    selectors = [lambda event_info : event_info.triggerType == "FORCE"]

    if len(config["run_time_range"]) == 0:
        run_time_range = None
    else:
        run_time_range = config["run_time_range"]

    # pure mattak
    mattak_ds = mattak.Dataset.Dataset(station=args.station, run=args.run, data_path= data_dir, backend="uproot")
    mattak_ds.setEntries(9)
    mattak_wfs = mattak_ds.wfs(calibrated=True)   
    print(mattak_wfs.shape)

    # NuRadio
    mattak_kw = dict(backend = "uproot", read_daq_status = False)
    rnog_reader.begin(root_dirs,
                      selectors=selectors,
                      read_calibrated_data=True,
                      apply_baseline_correction="approximate",
                      convert_to_voltage=False,
                      select_runs=True,
                      run_types=["physics"],
                      run_time_range=run_time_range,
                      max_trigger_rate=2 * units.Hz,
                      mattak_kwargs=mattak_kw)
    wfs = []

    for event in rnog_reader.run():
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        print(station.get_triggers())
        print(event.get_run_number())
        
        i = 0
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            wfs.append(channel.get_trace())
            i += 1
       
        break

    wfs = np.array(wfs)
    print(wfs.shape)


    pp = PdfPages("test_vc.pdf")

    fig_mattak, axs = plt.subplots(4, 6, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)
    for i, ax in enumerate(axs):
        ax.plot(mattak_wfs[i])
    fig_mattak.suptitle("Mattak ds (definitely in V)")
    pp.savefig(fig_mattak)

    fig_nuradio, axs = plt.subplots(4, 6, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)
    for i, ax in enumerate(axs):
        ax.plot(wfs[i])
    pp.savefig(fig_nuradio)

    pp.close()
    print(units.mV)
