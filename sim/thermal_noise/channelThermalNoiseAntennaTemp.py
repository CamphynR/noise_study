"""
This module adds thermal noise to channel trace by drawing from Rayleigh distributions
with scale factor calculated from the antenna temperatures. Antenna temperature calculations were taken from
Steffen Hallman's work and integrate the surrounding ice temperature with the antenna gain. 
"""

import argparse
import logging
import numpy as np

from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.channelThermalNoiseAntennaTemp")


class channelThermalNoiseAntennaTemp():
    def __init__(self, log_level=logging.DEBUG):
        logger.setLevel(log_level)
        pass

    def begin(self):
        pass


    def run(self, event, station, detector):
        pass
