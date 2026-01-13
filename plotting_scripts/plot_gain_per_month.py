import argparse
from astropy.time import Time
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pickle
import scipy.interpolate as interpolate

from NuRadioReco.utilities import units

from fitting.spectrumFitter import spectrumFitter
from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    return frequencies, frequency_spectrum

def read_time(path):
    result_dictionary = read_pickle(path)
    begin_time = result_dictionary["header"]["begin_time"][0]
    end_time = result_dictionary["header"]["end_time"][0]
    return np.mean([end_time.unix, begin_time.unix])

def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db

def convert_error_to_db(gain_error, gain):
    db_error = 20 * 1/(np.log(10) * gain) * gain_error
    return db_error


def read_vrms_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    vrms = result_dictionary["vrms"]
    var_vrms = result_dictionary["var_vrms"]
    nr_events = header["nr_events"]
    return vrms, var_vrms, nr_events, header


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", nargs="+")
    parser.add_argument("-v", "--vrms", nargs="+")
    parser.add_argument("-s", "--sims", nargs="+")
    parser.add_argument("-c", "--channels", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    channel_ids = np.arange(24)

    config_path = find_config(args.sims[0], sim=True)
    config = read_config(config_path)
    station_id = config["station"]
    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]

    times = []
    gains = [[] for i in channel_ids]
    gain_errors = [[] for i in channel_ids]
    vrms_list = []
    vrms_errors = []

    args.data = natsorted(args.data)
    args.vrms = natsorted(args.vrms)
    for i, data in enumerate(args.data):
        print(f"fitting {i+1}/{len(args.data)}")
        time = read_time(data)
        times.append(time)
        spectrum_fitter = spectrumFitter(data, args.sims)
        fit_results = spectrum_fitter.get_fit_gain(mode="electronic_temp", choose_channels=channel_ids)
        for channel_id in channel_ids:
            fit_params = fit_results[channel_id]
            gains[channel_id].append(fit_params[0].value)
            gain_errors[channel_id].append(fit_params[0].error)
        if args.test:
            if i ==10:
                break
        print(Time(time, format="unix").to_datetime().strftime("%B"))
        vrms, vrms_error, _, _ = read_vrms_file(args.vrms[i])
        vrms_list.append(vrms)
        vrms_errors.append(vrms_error)

    vrms_list = np.array(vrms_list).T
    vrms_errors = np.array(vrms_errors).T

    gains = np.array(gains)
    gain_errors = np.array(gain_errors)

    times_date = [Time(t, format="unix").to_datetime().strftime("%B") for t in times]
    season = Time(times[0], format="unix").to_datetime().strftime("%Y")
    print(season)
    gain_errors_db = convert_error_to_db(gain_errors, gains)
    gains_db = convert_to_db(gains)
     
    plt.style.use("gaudi")
    figname = f"figures/absolute_ampl_calibration/gains_per_month_season{season}_st{station_id}.pdf"
    if args.channels is not None:
        figname += "_ch_" + "_".join(str(ch_id) for ch_id in channel_ids)
    pdf = PdfPages(figname)
    for channel_id in channel_ids:
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        gain_db_line = axs[0].errorbar(times, gains_db[channel_id], yerr=gain_errors_db[channel_id],
                                       label = "gain with HESSE fit error")
        axs[0].set_ylabel("gain / dB")
        axs[0].set_title(f"Channel {channel_id}")
        axs[0].legend()

        axs[1].errorbar(times, vrms_list[channel_id] / units.mV, yerr=vrms_errors[channel_id] / units.mV)
        axs[1].set_ylabel("vrms / mV")
        axs[1].set_xticks(times, times_date, rotation=45)
        
        fig.suptitle(f"Station {station_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()
    pdf.close()
   
    
