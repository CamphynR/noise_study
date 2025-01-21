"""
Contains all theoretical functions to convert a temperature to noise
"""

import numpy as np

from scipy import constants
from NuRadioReco.utilities import units


def temp_to_volt(temp, bandwidth):
    """
    bandwidth: int
        frequency bandwidth of simulated noise
        expected to be in GHz

    returns
    -------
    voltage: float
        voltage in NuRadio base units
    """
    # check where this value comes from, it's often used in RNO-G for this formula to describe electronic noise
    resistance = 50 * units.ohm
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    voltage = np.sqrt(4 * k * temp * resistance * bandwidth)
    return voltage


