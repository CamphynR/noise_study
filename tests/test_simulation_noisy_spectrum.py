import argparse
from astropy.time import Time
import datetime
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
from scipy import constants
import subprocess

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units, fft

from modules.channelGalacticSunNoiseAdder import channelGalacticSunNoiseAdder
from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator

from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir


class dummyDet:
    def __init__(self):
        pass
    def get_component(self, collection="dummy", component="dummy_cmp"):
        return lambda freq : 1
    def get_detector_time(self):
        return Time("1900-01-01")


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


def padding_function(trace):
    padded_trace = np.pad(trace, len(trace), mode="reflect")
    filt = np.r_[np.linspace(0., 1., len(trace)/2), np.ones_like(trace), np.linspace(1., 0., len(trace)/2)]
    padded_trace *= filt
    return padded_trace


def create_sim_event(station_id, channel_ids, detector, frequencies, sampling_rate):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time(detector.get_detector_time())
    for channel_id in channel_ids:
        channel = Channel(channel_id)
        channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
        station.add_channel(channel)
    event.set_station(station)
    return event


def get_traces_from_event(event):
    station = event.get_station()
    traces = []
    for channel in station.iter_channels():
        trace = channel.get_trace()
        traces.append(trace)
    return np.array(traces)



def create_thermal_noise_events(nr_events, station_id, detector,
                                config,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                noise_sources=["ice", "electronic", "galactic"],
                                include_sum = True,
                                electronic_temperature=80*units.kelvin,
                                passband = None,
                                padding_length=8192):
    # electronic noise temperature, refer to eric's POS for this (PoS(ICRC2023)1171)

    # detector and trace parameters
    channel_ids = choose_channels 
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)


    # arbitrary choice but should be wide enough to incorporate detectors frequency fov
    min_freq = 10 * units.MHz
    max_freq = 1600 * units.MHz
    resistance = 50 * units.ohm
    amplitude = temp_to_volt(electronic_temperature, min_freq, max_freq, frequencies, resistance,
                             filter_type="rectangular")


    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()

    galactic_noise_adder = channelGalacticNoiseAdder()
    galactic_noise_adder.begin(freq_range=[min_freq, max_freq],
                               caching=True)

    system_response = systemResonseTimeDomainIncorporator()
    system_response.begin(det=detector)


    events = []
    for _ in range(nr_events):
        nr_event_types = len(noise_sources)
        if include_sum:
            nr_event_types += 1
        event_types = [create_sim_event(station_id, channel_ids, detector,
                                        frequencies, sampling_rate) 
                       for event_type in range(nr_event_types)]
        
        for i, noise_source in enumerate(noise_sources):
            station = event_types[i].get_station()
            if noise_source == "electronic":
                generic_noise_adder.run(event_types[i], station, detector, amplitude=amplitude, min_freq=min_freq, max_freq=max_freq, type="rayleigh")
            elif noise_source == "galactic":
                galactic_noise_adder.run(event_types[i], station, detector)
            elif noise_source == "galactic_min":
                time = datetime.datetime(2023, 8, 13, 11)
                galactic_noise_adder.run(event_types[i], station, detector, manual_time=time)
            elif noise_source == "galactic_max":
                time = datetime.datetime(2023, 8, 13, 23)
                galactic_noise_adder.run(event_types[i], station, detector, manual_time=time)


        if include_det_signal_chain:
            for event in event_types:
                station = event.get_station()
#                hardware_response.run(event, station, det=detector, sim_to_data=True)
                system_response.run(event, station, detector)
        events.append(event_types)
    
    events = np.array(events)
    # switch nr_events and noise_source indices
    events = events.T
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", default=23, type=int)
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    args = parser.parse_args()

    config = read_config(args.config)

    log_level = logging.WARNING 
    logger = logging.getLogger("NuRadioReco")
    logger.setLevel(log_level)

    station_id = args.station

    noise_sources = ["electronic"]
    include_sum = False
    electronic_temperature = 80 * units.kelvin

    
    channels_to_include = [0, 19]
    events_per_batch = 2000
    dummy_det = dummyDet()
    events = create_thermal_noise_events(events_per_batch, args.station, dummy_det, config,
                                         choose_channels = channels_to_include,
                                         include_det_signal_chain=config["include_det_signal_chain"],
                                         noise_sources=noise_sources, include_sum=include_sum,
                                         electronic_temperature=electronic_temperature,
                                         passband=[10 * units.MHz, 1600 * units.MHz]
                                         )

    plt.style.use("gaudi")
    labels = ["electronic"]
    fig, ax = plt.subplots()
    freq_spectra = []

    for i, event in enumerate(events[0]):
        station = event.get_station()
        channel = station.get_channel(19)
        nr_samples = channel.get_number_of_samples()
        sampling_rate = channel.get_sampling_rate()
        frequencies = np.fft.rfftfreq(nr_samples, d = 1./sampling_rate)
        frequency_spectrum = channel.get_frequency_spectrum()
        freq_spectra.append(frequency_spectrum)
    frequency_spectrum_mean = np.mean(np.abs(freq_spectra), axis=0)

    ax.plot(frequencies, frequency_spectrum_mean)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("Spectral amplitude / V/GHZ")
    ax.set_xlim(None, 1.)
    fig.savefig("test_sim.png")
