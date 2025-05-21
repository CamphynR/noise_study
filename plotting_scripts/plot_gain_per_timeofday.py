import argparse
from astropy.time import Time
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
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

def read_datetime(path):
    result_dictionary = read_pickle(path)
    begin_time = result_dictionary["header"]["begin_time"][0]
    end_time = result_dictionary["header"]["end_time"][0]
    return begin_time

def read_timeofday(path):
    result_dictionary = read_pickle(path)
    begin_time = result_dictionary["header"]["begin_time"][0]
    time_of_day_sec = begin_time.datetime.hour * 3600 \
                    + begin_time.datetime.minute * 60 \
                    + begin_time.datetime.second
    return time_of_day_sec

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
    absolute_times = []
    gains = [[] for i in channel_ids]
    gain_errors = [[] for i in channel_ids]
    month_indices = []
    prev_month = -1
    for i, data in enumerate(args.data):
        print(f"fitting {i+1}/{len(args.data)}")
        time = read_timeofday(data)
        times.append(time)
        absolute_time = read_datetime(data)
        month = absolute_time.datetime.month
        if month != prev_month:
            month_indices.append(i)
            prev_month = month
        absolute_times.append(absolute_time)
        spectrum_fitter = spectrumFitter(data, args.sims, bandpass=[0.01, 0.9])
        fit_results = spectrum_fitter.get_fit_gain(mode="electronic_temp", choose_channels=channel_ids)
        for j, channel_id in enumerate(channel_ids):
            fit_params = fit_results[j]
            gains[j].append(fit_params[2].value)
            gain_errors[j].append(fit_params[2].error)
        if args.test:
            if i ==10:
                break
    month_indices.append(-1)

    gains = np.array(gains)
    gain_errors = np.array(gain_errors)

    times_date = [Time(t, format="unix").to_datetime().strftime("%H") for t in times]
    gain_errors_db = convert_error_to_db(gain_errors, gains)
    gains_db = convert_to_db(gains) 

    start_month = min(absolute_times).datetime.strftime("%B")
    end_month = max(absolute_times).datetime.strftime("%B")

#    temp_path = glob.glob(f"station_temperatures/season2023/housekeepingdata_st{station_id}_*")[0]
#    temp_time, temperature = read_temperatures(temp_path, times)

    day_sec = 24 * 60 * 60
    xticks = np.arange(0, day_sec, 3600)

    # from https://www.timeanddate.com/sun/greenland/summit-camp?month=4&year=2023
    # in April at Summit
    sunrise_earliest = 2*60*60 + 22*60
    sunrise_latest = 5*60*60 + 6*60
    sunset_earliest = 20*60*60 + 10*60
    sunset_latest = 22*60*60 + 47*60

    plt.style.use("retro")

    sunrise_earliest_rectangle = Rectangle((-3600., 0.), sunrise_earliest, 1.5*np.max(gains_db), color="lightsteelblue", alpha=0.5)
    sunrise_latest_rectangle = Rectangle((-3600., 0.), sunrise_latest, 1.5*np.max(gains_db), color="lightsteelblue", alpha=0.5)
    sunset_earliest_rectangle = Rectangle((sunset_earliest, 0.), 24*60*60, 1.5*np.max(gains_db), color="lightsteelblue", alpha=0.5)
    sunset_latest_rectangle = Rectangle((sunset_latest, 0.), 24*60*60, 1.5*np.max(gains_db), color="lightsteelblue", alpha=0.5)
 
    figname = f"figures/absolute_ampl_calibration/gains_per_timeofday_st{station_id}"
    if args.channels is not None:
        figname += "_ch_" + "_".join(str(ch_id) for ch_id in channel_ids)
    figname += ".pdf"
    pdf = PdfPages(figname)
    for j, channel_id in enumerate(channel_ids):
        labels = iter(np.unique([t.datetime.strftime("%B") for t in absolute_times]))
        fig, ax = plt.subplots(figsize=(10, 10))
        for istart,iend in zip(month_indices[:-1], month_indices[1:]):
            ax.errorbar(times[istart:iend], gains_db[j][istart:iend],
                        yerr=gain_errors_db[j][istart:iend],
                        label=next(labels),
                        fmt="o")
#        ax.add_artist(sunrise_earliest_rectangle)
#        ax.add_artist(sunrise_latest_rectangle)
#        ax.add_artist(sunset_earliest_rectangle)
#        ax.add_artist(sunset_latest_rectangle)
        ax.legend()
#        ax.set_ylim(None, 47)

        ax.set_xticks(xticks, labels=[f"{i:02}" for i in np.arange(24)], rotation=-50)
        ax.set_xlabel("time of day / hour")
        ax.set_ylabel("gain / dB")
        ax.set_title(f"Channel {channel_id}")
        fig.suptitle(f"Station {station_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        if args.vector_image:
           fig.savefig(figname.split("ch")[0] + f"ch_{channel_id}.png", format="png", bbox_inches="tight") 
           fig.savefig(figname.split("ch")[0] + f"ch_{channel_id}.svg", format="svg", bbox_inches="tight") 
        plt.close()
    pdf.close()
