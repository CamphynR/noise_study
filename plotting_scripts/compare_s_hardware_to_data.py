import argparse
from astropy.time import Time
import csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os


from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator

from utilities.utility_functions import read_pickle


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return {"frequencies": frequencies,
            "spectrum" : frequency_spectrum,
            "var_spectrum" : var_frequency_spectrum,
            "header" : header}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--sims", nargs = "+")
    args = parser.parse_args()


    station_id = 24




    data = read_freq_spec_file(args.data)
    ice_sim = read_freq_spec_file(args.sims[0])
    electronic_sim = read_freq_spec_file(args.sims[1])
    galactic_sim = read_freq_spec_file(args.sims[2])
    sim_spectrum = ice_sim["spectrum"] + electronic_sim["spectrum"] + galactic_sim["spectrum"]



    det = Detector(source="rnog_mongo",
                   select_stations=station_id)
    det_time = Time("2023-08-01")
    det.update(det_time)

    nr_channels = det.get_number_of_channels(station_id)
    sampling_rate = det.get_sampling_frequency(station_id)
    nr_samples = det.get_number_of_samples(station_id)
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    bandpass = channelBandPassFilter()
    filt = np.abs(bandpass.get_filter(frequencies, station_id=0, channel_id=0, det=0, passband=[0.1, 0.7], filter_type="butter", order=10))


    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()


    plt.style.use("retro")
    pdf = PdfPages(f"figures/response_comparisons/compare_s_hardware_vs_data.pdf") 
    for channel_id in range(nr_channels):
        s_param_response = filt * np.abs(hardware_response.get_filter(frequencies, station_id, channel_id,
                                                                      det, sim_to_data=True))

        fig, ax = plt.subplots()
        ax.plot(frequencies, sim_spectrum[channel_id] * s_param_response, label="Simulation", lw=2.)
        ax.plot(frequencies, data["spectrum"][channel_id], label="Data", lw=2.)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectra amplitude V/GHz")
        ax.set_title(f"Station {station_id} channel {channel_id}")
        ax.set_xlim(0, 1.)
        ax.minorticks_on()
        ax.grid(True, which="minor", ls="dashed", color="gray", alpha=0.4)
        ax.legend()
        plt.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()
