from NuRadioReco.detector import detector

class detectorFilter():
    def __init__(self):
        pass

    def begin(self, det : detector.Detector, nr_of_channels = 24):
        self._det = det
        self._station_ids = det.get_station_ids()
        if isinstance(self._station_ids, int):
            self._station_ids = [self._station_ids]

        self.det_response = [[[] for i in range(nr_of_channels)] for s in self._station_ids]
        for station_id in self._station_ids:
            for channel_id in range(nr_of_channels):
                self._det_response[station_id, channel_id] = det.get_signal_chain_response(station_id, channel_id)

    def run(self, event, station):
        pass

    def end():
        pass

    def __str__(self):
        print(f"filter contains info on stations {self._station_ids}")
        print(self.det_response)

# test
if __name__ == "__main__":
    import os
    import glob
    import logging
    import datetime
    import argparse
    import matplotlib.pyplot as plt
   
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    from plotting_functions import plot_ft

    parser = argparse.ArgumentParser(prog = "%(prog)s", usage = "detector filter test")
    parser.add_argument("--station", type = list, default = [24], nargs = "?")
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

    detectorFilter = detectorFilter()
    detectorFilter.begin(det = det)
    print(detectorFilter())
    print(detectorFilter.detetcor_response())

    # for event in rnog_reader.run():
    #     station_id = event.get_station_ids()[0]
    #     station = event.get_station(station_id)

    #     fig, ax = plt.subplots()
    #     plot_ft(station.get_channel(0), ax, label = "before")
    #     cwFilter.run(event, station)
    #     plot_ft(station.get_channel(0), ax, label = "after")
    #     fig_dir = os.path.abspath("{__file__}/../figures")
    #     fig.savefig(f"{fig_dir}/test_detector_filter")
    #     break