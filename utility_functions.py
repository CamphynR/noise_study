import pickle
import numpy as np
from scipy import constants
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units

# Coding helper functions

def open_pickle(pickle_file):
    with open(pickle_file, "rb") as file:
        content = pickle.load(file)
    return content



# Theoretical functions

def reduce_by_db(variable, db):
    db_coeff = 10**(-1*db/20.)
    variable = db_coeff * variable
    return variable


def rms_from_temp(noise_temp_channel, detector, station, channel, include_amps=False):
    """
    This is Nyquist noise assuming a resistance of 50 Ohm, the bandpass in the formula
    is derived from the detector response and includes the amplifiers
    """
    freqs = np.linspace(10, 1000, 1000) * units.MHz
    # the analysis uses a bandpassfilter on the noise rms
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    bandpassfilter = channelBandPassFilter.get_filter(
            freqs, station, channel, detector, passband=[200*units.MHz, 600*units.MHz],
            filter_type='rectangular')

    response = detector.get_signal_chain_response(station, channel)
    filt = response(freqs)
    filt = np.convolve(filt, bandpassfilter, mode="same") 
    # convolve with bandpass filter
    if not include_amps:
        filt = filt / np.abs(filt).max()
    integrated_channel_response = np.trapezoid(np.abs(filt) ** 2, freqs)

    Vrms = (50 * noise_temp_channel * constants.k * integrated_channel_response / units.Hz)**0.5
    return Vrms



