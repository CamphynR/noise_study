import datetime
import logging
import glob
import numpy as np
import pickle
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

import config

if __name__ == "__main__":
    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = config.stations)
    det.update(datetime.datetime(2022, 8, 1))

    rnog_reader = readRNOGData(log_level = logging.DEBUG) #note if no runtable provided, runtable is queried from the database
    if len(config.stations) == 1:
        root_dirs = glob.glob(f"{config.data_dir}/station{config.stations[0]}/run*/")[:1]
        print(root_dirs)
    selectors = lambda event_info : event_info.triggerType == "FORCE"
    mattak_kw = dict(backend = "uproot", read_daq_status = False)
    rnog_reader.begin(root_dirs,    
                      selectors = selectors,
                      read_calibrated_data = False,
                      apply_baseline_correction="approximate",
                      convert_to_voltage = False,
                      select_runs = True,
                      run_types = ["physics"],
                      max_trigger_rate = 2 * units.Hz,
                      mattak_kwargs = mattak_kw)
    
    rms_list = [[] for i in range(24)]
    for i_event, event in enumerate(rnog_reader.run()):
        station_id = event.get_station_ids()[0]
        print(station_id)   
        station = event.get_station(station_id)
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            trace = channel.get_trace()
            rms = np.sqrt(np.mean(trace**2))
            print(rms)
            rms_list[channel_id].append(rms)

    # with open(f"rms_lists/rms_s{config.stations[0]}.pickle", "wb") as f:
    #     pickle.dump(rms_list, f)