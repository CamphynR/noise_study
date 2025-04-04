"""
Contains all theoretical functions to convert a temperature to noise
"""

import numpy as np

from scipy import constants
from NuRadioReco.utilities import units


def temp_to_volt_naive(temperature, bandwidth):
    """
    to see why factor 4 dissapears see wiki alinea "maximum power transfer" in Johnson-Nyquist noise page
    parameters
    ----------
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
    vrms = np.sqrt(k * temperature * resistance * bandwidth)
    return vrms


def temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance=50*units.ohm, filter_type="rectangular"):
    if filter_type=="rectangular":
        filt = np.zeros_like(frequencies)
        filt[np.where(np.logical_and(min_freq < frequencies , frequencies < max_freq))] = 1
    else:
        print("Other filters not yet implemented")
    bandwidth = np.trapz(np.abs(filt)**2, frequencies)
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    vrms = np.sqrt(k * temperature * resistance * bandwidth)
    return vrms
