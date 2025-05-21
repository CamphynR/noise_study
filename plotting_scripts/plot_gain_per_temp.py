import argparse
from astropy.time import Time
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
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


def create_temp_function(temp_path):
    time = []
    temperature = []
    with open(temp_path, "r") as temp_file:
        reader = csv.DictReader(temp_file)
        for i, row in enumerate(reader):
            time_tmp = float(row["time [unix s]"])
            time.append(time_tmp)
            #remote 1 data
            try:
                temperature.append(float(row["heat [\N{DEGREE SIGN}C]"]))
            # weather data
            except:
                temperature.append(float(row["temperature [\N{DEGREE SIGN}C]"]))
    return interpolate.interp1d(time, temperature, bounds_error=False, fill_value=0.)


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
    parser.add_argument("--channels", nargs="+", default=None, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--vector_image", action="store_true")
    args = parser.parse_args()

    
    config_path = find_config(args.sims[0], sim=True)
    config = read_config(config_path)
    station_id = config["station"]
    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]


    
    if args.channels is None:
        channel_ids = np.arange(24)
    else:
        channel_ids = args.channels
    times = []
    gains = [[] for i in channel_ids]
    gain_errors = [[] for i in channel_ids]
    for i, data in enumerate(args.data):
        print(f"fitting {i+1}/{len(args.data)}")
        time = read_time(data)
        times.append(time)
        spectrum_fitter = spectrumFitter(data, args.sims)
        fit_results = spectrum_fitter.get_fit_gain(mode="electronic_temp", choose_channels=channel_ids)
        for j, channel_id in enumerate(channel_ids):
            fit_params = fit_results[j]
            gains[j].append(fit_params[0].value)
            gain_errors[j].append(fit_params[0].error)
        if args.test:
            if i ==10:
                break

    gains = np.array(gains)
    gain_errors = np.array(gain_errors)

    times_date = [Time(t, format="unix").to_datetime().strftime("%Y-%B-%d") for t in times]
    gain_errors_db = convert_error_to_db(gain_errors, gains)
    gains_db = convert_to_db(gains)
    

    temp_path = glob.glob(f"station_temperatures/weather/season2023/weatherdata_st{station_id}_*")[0]
    temp_func = create_temp_function(temp_path)
    temperature = temp_func(times)
    plt.style.use("gaudi")
    plt.scatter(temperature, gains_db[0], label="runs")
    plt.legend()
    plt.xlabel("temperature / \N{DEGREE SIGN} C")
    plt.ylabel("gain / dB")
    plt.title(f"Station {station_id}, channel 0")
    plt.savefig("test_temp_gain")


    plt.style.use("retro")
    figname = f"figures/absolute_ampl_calibration/temp_gain_correlation_st{station_id}"
    if args.channels is not None:
        figname += "_ch_" + "_".join(str(ch_id) for ch_id in channel_ids)
    figname += ".pdf"
    pdf = PdfPages(figname)
    for j, channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.errorbar(temperature, gains_db[j], yerr=gain_errors_db[j], fmt="o",
                    label="run", markersize=5, linewidth=1.5)
        correlation = np.corrcoef(temperature, gains_db[j])
        ax.text(0.05, 0.05, f"Corr = {correlation[0, 1]:.2f}",
                transform=ax.transAxes,
                bbox={
                    "boxstyle": "round",
                    "facecolor" : "white",
                    "edgecolor" : "black"
                    })
        ax.set_xlabel("weather temperature at Summit / \N{DEGREE SIGN}C")
        ax.set_ylabel("gain / dB")
        
        ax_title = ax.set_title(f"Channel {channel_id}")
        ax.minorticks_on()
        ax.grid(which="minor", visible=True, alpha=0.15, linewidth=0.25, color="gray")
        ax.set_axisbelow(True)
        ax.legend()

        mid = (fig.subplotpars.right + fig.subplotpars.left)/2
        height = 1.1 * fig.subplotpars.top
        fig.suptitle(f"Station {station_id}", x=mid)
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        if args.vector_image:
            fig.savefig(figname.split("ch")[0] + f"ch_{channel_id}.svg", format="svg", bbox_inches="tight") 
            fig.savefig(figname.split("ch")[0] + f"ch_{channel_id}.png", format="png", bbox_inches="tight") 
        plt.close()
    pdf.close()
