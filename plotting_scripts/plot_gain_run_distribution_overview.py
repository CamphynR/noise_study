import argparse
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from scipy import interpolate, stats

from utilities.utility_functions import convert_to_db, convert_error_to_db, read_pickle


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    args = parser.parse_args()

    station_ids = [11, 12, 13, 21, 23, 24]
    channel_ids = range(24)

    gain_run_folders = natsorted([f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/system_amplitude_calibration/season{args.season}/station{station}/results_per_run/" for station in station_ids])
    files = []
    for gain_run_folder in gain_run_folders:
        files_station = os.listdir(gain_run_folder)
        files_station = [f for f in files_station if not f.endswith("error.csv")]
        files.append(files_station)


    runs = []
    dfs = []
    for i, files_station in enumerate(files):
        runs_station = []
        dfs_station = []
        for file in files_station[::20]:
            run = int(file.split("run")[1].split(".csv")[0])
            runs_station.append(run)
            df = pd.read_csv(gain_run_folders[i] + file)
            df.rename(columns={"Unnamed: 0" : "channel"}, inplace=True)
            df.insert(0, "run", run)
            dfs_station.append(df)
        runs.append(runs_station)
        dfs_station = pd.concat(dfs_station, axis=0)
        dfs.append(dfs_station)



    confidence_intervals = []
    for df in dfs:
        confidence_intervals_station = []
        for channel_id in channel_ids:
            confidence=0.95

            gains = convert_to_db(df[df["channel"] == channel_id]["gain"].to_numpy())
            mean, sigma = np.mean(gains), np.std(gains, ddof=1)
            confidence_interval = stats.norm.interval(confidence, loc=mean, scale=sigma)
            confidence_interval = np.diff(confidence_interval)
            confidence_intervals_station.append(confidence_interval[0])
        confidence_intervals.append(confidence_intervals_station)
    confidence_intervals = np.array(confidence_intervals)

    deep_channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
    pdf = PdfPages(f"figures/variability/gain_run_distribution_overview_season{args.season}.pdf")
    plt.style.use("retro")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, ax = plt.subplots()
    ax.violinplot(confidence_intervals[:, deep_channels].T, vert=True, showmeans=True)
    for channel_id in deep_channels:
        ax.scatter(np.arange(1, len(station_ids)+1), confidence_intervals[:, channel_id], label=f"Channel {channel_id}")
    ax.set_xticks(np.arange(1, len(station_ids)+1), station_ids)
    
    ax.set_xlabel("Station")
    ax.set_ylabel("Distribution spread / dB")
    ax.legend(loc="upper left", bbox_to_anchor=(1.,1.), ncols=2)
    ax.set_title(f"Season {args.season}")
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    pdf.close()
    exit()

    
    average_ft_run_folder = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season{args.season}/station{args.station}/clean/"
    times = []
    for run in df["run"].unique():
        path = average_ft_run_folder + f"average_ft_run{run}.pickle"
        time = read_time(path)
        times.append(time)
    
    temp_path = glob.glob(f"station_temperatures/remote1/season{args.season}/housekeepingdata_st{args.station}_*")[0]
    temp_func = create_temp_function(temp_path)
    temperature = temp_func(times)
    

    pdf_temp = PdfPages(f"figures/variability/gain_vs_temp_season{args.season}_st{args.station}.pdf")

    for channel_id in channel_ids:
        gains = convert_to_db(df[df["channel"] == channel_id]["gain"].to_numpy())


        fig, ax = plt.subplots()
        ax.scatter(temperature, gains)
        ax.set_xlabel("temperature / C (remote 1)")
        ax.set_ylabel("Gain / dB")
        ax.set_title(f"Gain and temperature per run station {args.station} channel {channel_id}")
        fig.savefig(pdf_temp, format="pdf")
        plt.close(fig)
    pdf_temp.close()
