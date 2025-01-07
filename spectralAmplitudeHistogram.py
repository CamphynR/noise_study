import glob
import os

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

from utility_functions import read_config


class spectralAmplitudeHistogram():
    """
    wrapper class for the NuRadio data parser to specifically return histograms
    of spectral amplitudes
    """

    def __init__(self, log_level):
        self.reader = readRNOGData(log_level=log_level)
        return
    
    def begin(self, station, config_path):
        self.station = station
        self.config = read_config(config_path)

        if not self.data_dir:
            self.data_dir = os.environ["RNO_G_DATA"]
            self.root_dirs = glob.glob(f"{self.data_dir}/station{self.station}/run*")

        self.reader.begin()
        return
        
    def parse(self):
        pass

    def set_data_dir(self, data_dir):
        self.data_dir = data_dir
        return
