import json
import numpy as np
import os
from pathlib import Path
import pickle
from scipy import constants

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.utilities import units
import NuRadioMC.utilities.medium as medium

# simulation functions

def create_sim_event(station_id, channel_id, detector, frequencies, sampling_rate):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time(detector.get_detector_time())
    channel = Channel(channel_id)
    channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
    station.add_channel(channel)
    event.set_station(station)
    return event, station


# Coding helper functions

def read_pickle(pickle_file):
    with open(pickle_file, "rb") as file:
        content = pickle.load(file)
    return content

def write_pickle(data, pickle_file):
    with open(pickle_file, "wb") as file:
        pickle.dump(data, file)
    return


def find_config(data_dir, sim=False):
    """
    function that yields config file, assuming the config to be stored in the stationX folder.
    """
    job_folder = Path(data_dir).parents[1]
#    if sim:
#        job_folder = Path(data_dir).parents[2]
    return str(job_folder)+ "/" + "/config.json"


def read_config(config_path):
    with open(config_path, "r") as config_json:
        config = json.load(config_json)
    return config


def create_nested_dir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        if os.path.isdir(directory):
            pass
        else:
            raise SystemError("os was unable to construct data folder hierarchy")


def select_ice_model(config):
    model = config["propagation"]["ice_model"]
    if model == "greenland_simple":
        return medium.greenland_simple()
    else:
        raise KeyError(f"{model} not found, update select_ice_model or try a different model")


# Theoretical functions

def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db

def convert_error_to_db(gain_error, gain):
    db_error = 20 * 1/(np.log(10) * gain) * gain_error
    return db_error

def reduce_by_db(variable, db):
    db_coeff = 10**(-1*db/20.)
    variable = db_coeff * variable
    return variable


def compute_confidence_interval(samples, confidence=0.95):
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1) #ddof=1 to get unbiased variance (i.e. divide by n-1)
    score = 1-confidence
    confidence_left = sample_mean - (score * np.sqrt(sample_var))/np.sqrt(len(samples))
    confidence_right = sample_mean + (score * np.sqrt(sample_var))/np.sqrt(len(samples))
    return [confidence_left, confidence_right]

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



def read_freq_spectrum_from_pickle(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return {"frequencies" : frequencies,
            "spectrum" : frequency_spectrum,
            "var_spectrum" : var_frequency_spectrum,
            "header" : header}



def read_freq_spectrum_from_nur(files : list, return_phase=False):
    event_reader = eventReader()
    event_reader.begin(files)
    spec = []
    for event in event_reader.run():
        station = event.get_station()
        spec_channel = []
        for channel in station.iter_channels():
            frequency_spectrum = channel.get_frequency_spectrum()
            if not return_phase:
                spec_channel.append(np.abs(frequency_spectrum))
            else:
                spec_channel.append(frequency_spectrum)
        spec.append(spec_channel)
    return np.array(spec)
