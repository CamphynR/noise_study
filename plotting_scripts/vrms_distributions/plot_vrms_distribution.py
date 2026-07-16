import sys

import argparse
from astropy.time import Time
import copy
import glob
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from scipy import constants

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units, fft

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_pickle, read_config, read_freq_spectrum_from_nur



def temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance=50*units.ohm, filter_type="rectangular"):
    if filter_type=="rectangular":
        print(min_freq)
        print(frequencies)
        filt = np.zeros_like(frequencies)
        filt[np.where(np.logical_and(min_freq < frequencies , frequencies < max_freq))] = 1
    else:
        print("Other filters not yet implemented")
    bandwidth = np.trapz(np.abs(filt)**2, frequencies)
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    vrms = np.sqrt(k * temperature * resistance * bandwidth)
    return vrms


def generate_event(station_id, frequencies, channel_ids):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time("2023-08-01", format="isot")
    for channel_id in channel_ids:
        channel = Channel(channel_id)
        channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
        station.add_channel(channel)
    event.set_station(station)
    return event


def initialize_ice_adder():
    eff_temp_dir = "sim/library/eff_temperatures"
    ice_adder = channelThermalNoiseAdder()
    ice_adder.begin(sim_library_dir=eff_temp_dir)
    return ice_adder


def initialize_electronic_adder(min_freq, max_freq, frequencies, electronic_temp):
    resistance = 50 * units.ohm
    amplitude = temp_to_volt(electronic_temp, min_freq, max_freq, frequencies, resistance, filter_type="rectangular")
    electronic_adder = channelGenericNoiseAdder()

    return electronic_adder, amplitude


def initialize_galactic_adder(min_freq, max_freq):
    galactic_adder = channelGalacticNoiseAdder()
    galactic_adder.begin(freq_range=[min_freq, max_freq], caching=True)

    return galactic_adder


def add3components(event, station, det,
                   min_freq=10*units.MHz,
                   max_freq=1500*units.MHz,
                   electronic_temp=80*units.K):
    freqs = event.get_station().get_channel(0).get_frequencies()
    print(freqs.shape)
    ice_adder = initialize_ice_adder()
    electronic_adder, electronic_amp = initialize_electronic_adder(min_freq, max_freq, freqs, electronic_temp)
    galactic_adder = initialize_galactic_adder(min_freq, max_freq)

    ice_adder.run(event, station, det)
    electronic_adder.run(event, station, det, electronic_amp)
    galactic_adder.run(event, station, det)



def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return {"frequencies" : frequencies,
            "spectrum" : frequency_spectrum,
            "var_spectrum" : var_frequency_spectrum,
            "header" : header}



#def read_freq_spectrum_from_nur(files : list):
#    event_reader = eventReader()
#    event_reader.begin(files)
#    spec = []
#    for event in event_reader.run():
#        station = event.get_station()
#        spec_channel = []
#        for channel in station.iter_channels():
#            frequency_spectrum = channel.get_frequency_spectrum()
#            spec_channel.append(np.abs(frequency_spectrum))
#        spec.append(spec_channel)
#    return np.array(spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int)
    args = parser.parse_args()


    season = 2023
    station_id = args.station
    nr_channels = 24
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    channel_ids = np.arange(24)
#    channel_ids = [5]
    frequencies = fft.freqs(nr_samples, sampling_rate) 





#------------DATA-------------


    data_paths = natsorted(glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season{season}/station{station_id}/clean/average_vrms_run*.pickle"))

    times = []
    vrms = []
    var_vrms = []
    for pickle in data_paths:
        rms_dict = read_pickle(pickle)
        times.append(rms_dict["header"]["begin_time"].unix)
        vrms.append(rms_dict["vrms"])
        var_vrms.append(rms_dict["var_vrms"])
    vrms = np.array(vrms).T
    var_vrms = np.array(var_vrms).T

    times = np.array(times)[:, 0]
    times_date = [Time(t, format="unix").strftime("%Y-%B-%d") for t in times]


    

#----------------PLOTS------------------


    plt.style.use("astroparticle_physics")
    pdf = PdfPages(f"figures/vrms/vrms_scatter_season{season}_st{station_id}.pdf")
    for channel_id in channel_ids:
        fig, ax = plt.subplots()

        ax.scatter(times, vrms[channel_id, :])

        nr_ticks = 7
        tick_step = len(times) // nr_ticks
        ax.set_xticks(times[::tick_step], label=times_date[::tick_step])
        ax.set_xlabel("time")
        ax.set_ylabel("Vrms / mV")
        ax.set_xlabel("Vrms / mV")
        ax.set_ylabel("Counts")

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()



    pdf = PdfPages(f"figures/vrms/vrms_distribution_season{season}_st{station_id}.pdf")
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
#        hist, bin_edges = np.histogram(vrms[channel_id, :] / units.mV,
#                                       bins=50)
#        ax.bar(bin_edges[:-1], hist,
#               facecolor="white",
#               edgecolor="black",
#               linewidth=1.,
#               align="left")
        ax.scatter(vrms[channel_id, :] / units.mV, bins=50, histtype="step")

        ax.set_xlabel("Vrms / mV")
        ax.set_ylabel("Counts")

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()


    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPol" : [4, 8, 11, 21],
                     "LPDA up" : [13, 16, 19],
                     "LPDA down" : [12, 14, 15, 17, 19, 20]}

    for channel_type, channel_ids in channel_types.items():
        fig, ax = plt.subplots()
        ax.hist(np.ndarray.flatten(vrms[channel_ids]) / units.mV, bins=50, histtype="step")

        ax.set_xlabel("Vrms / mV")
        ax.set_ylabel("Counts")

        fig.tight_layout()
        figname = f"figures/vrms/vrms_distribution_season{season}_st{station_id}_{channel_type}.png"
        fig.savefig(figname, dpi=150)
50)
