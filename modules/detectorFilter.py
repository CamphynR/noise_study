import time
from NuRadioReco.detector import detector



class detectorFilter():
    def __init__(self):
        pass

    def begin(self, nr_of_channels = 24):
        self._nr_of_channels = 24
        
        # self._station_ids = det.get_station_ids()
        # if isinstance(self._station_ids, int):
        #     self._station_ids = [self._station_ids]
        # self._det_response = [[[] for i in range(self._nr_of_channels)] for s in range(len(self._station_ids))]
        # for idx, station_id in enumerate(self._station_ids):
        #     for channel_id in range(nr_of_channels):
        #         self.det_response[idx][channel_id] = det.get_signal_chain_response(station_id, channel_id)
        return
       
    def run(self, event, station, det : detector.Detector):
        t0 = time.time()
        channels_filtered = []
        for channel in station.iter_channels():
            station_id = station.get_id()
            channel_id = channel.get_id()
            self.det_response = det.get_signal_chain_response(station_id, channel_id)
            channels_filtered += [channel / self.det_response]

        for channel_id in range(self._nr_of_channels):
            station.remove_channel(channel_id)
            station.add_channel(channels_filtered[channel_id])
        self._time = time.time() - t0
        print(f"filtering station {station.get_id()} took {self._time}")

    def end():
        pass

    def __str__(self):
        return f"Detector deconvolution filter contains info on stations {self._station_ids}"



# test
if __name__ == "__main__":
    import os
    import glob
    import logging
    import numpy as np
    import datetime
    import argparse
    import matplotlib.pyplot as plt
   
    from NuRadioReco.utilities import units
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
    from plotting_functions import plot_ft

    parser = argparse.ArgumentParser(prog = "%(prog)s", usage = "detector filter test")
    parser.add_argument("--station", type = int, default = [24], nargs = "+")
    parser.add_argument("--run", type = int, default = 1)
    args = parser.parse_args()

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(2022, 7, 15))
    
    data_dir = os.environ["RNO_G_DATA"]
    rnog_reader = readRNOGData(log_level = logging.DEBUG)

    # multiple stations for checking the loading of the responses in begin()
    root_dirs = []
    for station_id in args.station:
        root_dirs += glob.glob(f"{data_dir}/station{station_id}/run{args.run}")
    

    rnog_reader.begin(root_dirs,
                      convert_to_voltage = True,
                      mattak_kwargs = dict(backend = "uproot"))

    channelBandPassFilter = channelBandPassFilter()
    channelBandPassFilter.begin()
    passband = [200 * units.MHz, 600 * units.MHz]

    detectorFilter = detectorFilter()
    detectorFilter.begin(det = det)
    print(detectorFilter)

    for event in rnog_reader.run():
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)

        fig, ax = plt.subplots()
        plot_ft(station.get_channel(0), ax, label = "before")

        channelBandPassFilter.run(event, station, det, passband = passband)
        detectorFilter.run(event, station)
        plot_kwargs = dict(c = "r", alpha = 0.5, ls = "dashed")
        plot_ft(station.get_channel(0), ax, label = "after", secondary = True, plot_kwargs = plot_kwargs)
        fig_dir = os.path.abspath("{__file__}/../figures")
        fig.savefig(f"{fig_dir}/test_detector_filter")
        break