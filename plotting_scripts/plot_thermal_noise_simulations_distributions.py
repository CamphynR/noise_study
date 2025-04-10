"""
Purely for testing
"""

import argparse
from astropy.time import Time
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import goodness_of_fit
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

from utilities.utility_functions import read_config, write_pickle 


def rayleigh(spec_amplitude, sigma):
    return (spec_amplitude / sigma**2) * np.exp(-spec_amplitude**2 / (2 * sigma**2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    sim_files = f"{config['save_dir']}/simulations/thermal_noise_traces"
    job_name = "job_2025_02_17_12_iceadder"
    sim_files = sim_files + "/" + job_name
#    ice_files = glob.glob(f"{sim_files}/ice_events*.nur")
#    electronic_files = glob.glob(f"{sim_files}/electronic_events*.nur")
    print(sim_files)
    sim_files = glob.glob(f"{sim_files}/station23/run*/events*.nur")
    print(sim_files)


    event_reader = eventReader()
    event_reader.begin(sim_files)

#    ice_event_reader = eventReader()
#    ice_event_reader.begin(ice_files)
#
#    electronic_event_reader = eventReader()
#    electronic_event_reader.begin(electronic_files)

    detector = detector.Detector(source="rnog_mongo", select_stations=23)
    detector_time = Time("2022-08-01")
    detector.update(detector_time)
    
    bandpass_filter = channelBandPassFilter()
    hardware_response = hardwareResponseIncorporator()

    hardware_response.begin()

    channel_ids = [0, 4]
    
    # very bruteforce way to get an idea of spectrum scale
    for event in event_reader.run():
        station = event.get_station()
        spec_max = 0
        for channel in station.iter_channels():
            if channel.get_id() not in channel_ids:
                continue
            frequencies = channel.get_frequencies()
            frequency_spectrum = channel.get_frequency_spectrum()
            spec_max_ch = np.max(np.abs(frequency_spectrum))
            if spec_max_ch > spec_max:
                spec_max = spec_max_ch
        break

    print(spec_max)
    

#    nr_bins = config["variable_function_kwargs"]["clean"]["nr_bins"]
#    hist_range = config["variable_function_kwargs"]["clean"]["hist_range"]
    nr_bins = 50
    hist_range = [0, spec_max]
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centres = bin_edges[:-1] + np.diff(bin_edges[0:2])/2


    spec_amplitude_histograms = np.zeros((24, len(frequencies), nr_bins))
    print("Starting read")
    for event in event_reader.run():
        station = event.get_station()
        station_id = station.get_id()

        bandpass_filter.run(event, station, det=None, passband = [200 * units.MHz, 600 * units.MHz])
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            frequencies = channel.get_frequencies()
            frequency_spectrum = channel.get_frequency_spectrum()


            bin_indices = np.searchsorted(bin_edges[1:-1], np.abs(frequency_spectrum))
            spec_amplitude_histograms[channel_id, np.arange(len(frequencies)), bin_indices] += 1

    # convert hists to pdf
    normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
    spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis],out=np.zeros_like(spec_amplitude_histograms), where=normalization_factor[:, :, np.newaxis]!=0)
    
#    spec_amplitude_histograms_ice = np.zeros((24, len(frequencies), nr_bins))
#    print("Starting read")
#    m = 0
#    for event in ice_event_reader.run():
#        station = event.get_station()
#        station_id = station.get_id()
#
#        bandpass_filter.run(event, station, det=None, passband = [200 * units.MHz, 600 * units.MHz])
#        for channel in station.iter_channels():
#            channel_id = channel.get_id()
#            frequencies = channel.get_frequencies()
#            frequency_spectrum = channel.get_frequency_spectrum()
#
#            bin_indices = np.searchsorted(bin_edges[1:-1], np.abs(frequency_spectrum))
#            spec_amplitude_histograms_ice[channel_id, np.arange(len(frequencies)), bin_indices] += 1
#
#    # convert hists to pdf
#    normalization_factor = np.sum(spec_amplitude_histograms_ice, axis = -1) * np.diff(bin_edges)[0]
#    spec_amplitude_histograms_ice = np.divide(spec_amplitude_histograms_ice, normalization_factor[:, :, np.newaxis],out=np.zeros_like(spec_amplitude_histograms), where=normalization_factor[:, :, np.newaxis]!=0)
#    
#    spec_amplitude_histograms_electronic = np.zeros((24, len(frequencies), nr_bins))
#    print("Starting read")
#    for event in electronic_event_reader.run():
#        station = event.get_station()
#        station_id = station.get_id()
#
#        bandpass_filter.run(event, station, det=None, passband = [200 * units.MHz, 600 * units.MHz])
#        for channel in station.iter_channels():
#            channel_id = channel.get_id()
#            frequencies = channel.get_frequencies()
#            frequency_spectrum = channel.get_frequency_spectrum()
#
#            bin_indices = np.searchsorted(bin_edges[1:-1], np.abs(frequency_spectrum))
#            spec_amplitude_histograms_electronic[channel_id, np.arange(len(frequencies)), bin_indices] += 1
#
#    # convert hists to pdf
#    normalization_factor = np.sum(spec_amplitude_histograms_electronic, axis = -1) * np.diff(bin_edges)[0]
#    spec_amplitude_histograms_electronic = np.divide(spec_amplitude_histograms_electronic, normalization_factor[:, :, np.newaxis],out=np.zeros_like(spec_amplitude_histograms_electronic), where=normalization_factor[:, :, np.newaxis]!=0)
    
    
    try:
        plt.style.use("~/envs/gaudi.mplstyle")
    except:
        pass

    fig_folder = f"{config['fig_dir']}/rayleigh"
    pdf_name = fig_folder + "/" + f"sim_rayleigh_param_s{station_id}_{job_name[4:]}" + ".pdf"     
    pdf = PdfPages(pdf_name)

    # sigma ifo frequency
    freq_range = [200*units.MHz, 600*units.MHz]
    frequency_indices = np.nonzero(np.logical_and(freq_range[0] < frequencies, frequencies < freq_range[1]))[0]

    sigma_guess = 0.01
    
    sigmas_save = []
    sigmas_save_ice = []
    sigmas_save_electronic = []
    covs_save = []
    for channel_id in channel_ids:
        print(channel_id)
        sigmas = np.zeros_like(frequencies)
        sigmas_ice = np.zeros_like(frequencies)
        sigmas_electronic = np.zeros_like(frequencies)
        covs = np.zeros_like(frequencies)
        for freq_idx in frequency_indices:
            sigma, cov = curve_fit(rayleigh, bin_centres, spec_amplitude_histograms[channel_id, freq_idx,:], p0=sigma_guess)
#            sigma_ice, cov = curve_fit(rayleigh, bin_centres, spec_amplitude_histograms_ice[channel_id, freq_idx,:], p0=sigma_guess)
#            sigma_electronic, cov = curve_fit(rayleigh, bin_centres, spec_amplitude_histograms_electronic[channel_id, freq_idx,:], p0=sigma_guess)
            sigmas[freq_idx] = sigma
#            sigmas_ice[freq_idx] = sigma_ice
#            sigmas_electronic[freq_idx] = sigma_electronic
            covs[freq_idx] = cov[0][0]
        sigmas_save.append(sigmas)
#        sigmas_save_ice.append(sigmas_ice)
#        sigmas_save_electronic.append(sigmas_electronic)
        covs_save.append(covs)
            
        fig, ax = plt.subplots()
        ax.plot(frequencies, sigmas, label = "sum")
#        ax.plot(frequencies, sigmas_ice, label = "ice")
#        ax.plot(frequencies, sigmas_electronic, label = "electronic")
        ax.legend()
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("sigma")
        ax.set_xlim(0.1, 0.7)
#        ax.set_ylim(0., 0.007)
        ax.set_title(f"station {station_id}, channel {channel_id}")
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()

    print("here")

    pickle_dict = {"freqs" : frequencies[frequency_indices], "sigmas" : sigmas_save, "covs" : covs_save}
    pickle_path = "data/rayleigh_parameters"
    pickle_path = pickle_path + "/" + job_name + ".pickle"
    write_pickle(pickle_dict, pickle_path)


    channel_id = 4
    # distributions at specific frequencies for testing purposes
    nr_freqs = 12
    fig, axs = plt.subplots(nr_freqs, 1, figsize = (12, 18), sharex=True)
    freqs = np.linspace(freq_range[0], freq_range[1], nr_freqs)
    for i , ax in enumerate(axs):
        freq_idx = np.where(np.isclose(freqs[i],frequencies, atol=np.diff(frequencies[0:2])))[0][0]
        sigma, cov = curve_fit(rayleigh, bin_centres, spec_amplitude_histograms[channel_id, freq_idx,:], p0=sigma_guess)
        ax.stairs(spec_amplitude_histograms[channel_id, freq_idx,:], edges=bin_edges)
        ax.plot(bin_centres, rayleigh(bin_centres, sigma), label = "rayleigh fit")
        ax.legend()
        ax.text(0.9, 0.9, f"sigma = {sigma}", va="top", ha="right", transform = ax.transAxes)
        ax.set_ylabel("N")
        ax.set_title(f"Distribution of simulated data, station {station_id}, freq = {freqs[i]:.2f} GHz", size="small")
    ax.set_xlabel("spectral amplitude / V/GHz")
    fig.tight_layout()
    fig_name = fig_folder + "/" + f"sim_rayleigh_distr_s{station_id}_{job_name[4:]}" + ".png"
    fig.savefig(fig_name)
