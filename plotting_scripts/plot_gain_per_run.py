import argparse
from astropy.time import Time
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.interpolate as interpolate

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


def read_temperatures(temp_path, selection_times = None):
    time = []
    temperature = []
    with open(temp_path, "r") as temp_file:
        reader = csv.DictReader(temp_file)
        for i, row in enumerate(reader):
            time_tmp = float(row["time [unix s]"])
            if np.logical_and(selection_times[0] < time_tmp, time_tmp < selection_times[-1]):
                time.append(time_tmp)
                temperature.append(float(row["heat [\N{DEGREE SIGN}C]"]))
    return time, temperature


def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db

def convert_error_to_db(gain_error, gain):
    db_error = 20 * 1/(np.log(10) * gain) * gain_error
    return db_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", nargs="+")
    parser.add_argument("--sims", "-s", nargs="+")
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    
    config_path = find_config(args.sims[0], sim=True)
    config = read_config(config_path)
    station_id = config["station"]
    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]


    spectrum_fitter = spectrumFitter(args.data[0], args.sims)
    season = spectrum_fitter.data_header["begin_time"].strftime("%Y")[0]
    save_folder=f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/system_amplitude_calibration/season{season}/station{station_id}/results_per_run"
    os.makedirs(save_folder, exist_ok=True) 

    
    if args.channels is None:
        channel_ids = np.arange(24)
    else:
        channel_ids = args.channels
    times = []
    gains = [[] for i in channel_ids]
    gain_errors = [[] for i in channel_ids]
    for i, data in enumerate(args.data):
        print(f"fitting {i+1}/{len(args.data)}")
        run = data.split("run", 1)[-1].split(".pickle")[0]
        print(run)
        time = read_time(data)
        times.append(time)
        spectrum_fitter = spectrumFitter(data, args.sims)
        filename = f"absolute_amplitude_calibration_season{season}_st{station_id}_run{run}.csv"
        fit_results = spectrum_fitter.save_fit_results(mode="electronic_temp",
                                                       save_folder=save_folder,
                                                       filename=filename, extended=True)
        for channel_id in channel_ids:
            fit_params = fit_results[channel_id]
            gains[channel_id].append(fit_params[0].value)
            gain_errors[channel_id].append(fit_params[0].error)
        if args.test:
            if i ==10:
                break

    gains = np.array(gains)
    gain_errors = np.array(gain_errors)

    times_date = [Time(t, format="unix").to_datetime().strftime("%Y-%B-%d") for t in times]
    gain_errors_db = convert_error_to_db(gain_errors, gains)
    gains_db = convert_to_db(gains)
    

    temp_path = glob.glob(f"station_temperatures/remote1/season2023/housekeepingdata_st{station_id}_*")[0]
    temp_time, temperature = read_temperatures(temp_path, times)

    nr_xticks = 10
    tick_step = int(len(times)/nr_xticks)


    
    plt.style.use("gaudi")
    figname = f"figures/absolute_ampl_calibration/gains_per_run_st{station_id}.pdf"
    if args.channels is not None:
        figname += "_ch_" + "_".join(str(ch_id) for ch_id in channel_ids)
    pdf = PdfPages(figname)
    for channel_id in channel_ids:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        gain_db_line = axs[0].errorbar(times, gains_db[channel_id], yerr=gain_errors_db[channel_id],
                                       label = "gain with HESSE fit error")
        axs[0].set_ylabel("gain / dB")
        axs[0].set_title(f"Channel {channel_id}")
        axs[0].legend()

        axs[1].plot(temp_time, temperature)
        axs[1].set_xticks(times[::tick_step], labels=times_date[::tick_step], rotation=-45)
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("temperature / \N{DEGREE SIGN}C")
        
        fig.suptitle(f"Station {station_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()
    pdf.close()


    #----------- Distributions ---------------------------------------------------

    print(gains.shape)
    fig, ax = plt.subplots()
    ax.hist(gains[0])
    plt.savefig("test")
