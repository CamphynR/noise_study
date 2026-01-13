import argparse
import csv
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
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
    parser.add_argument("--station", type=int, default=11)
    args = parser.parse_args()

    channel_ids = range(24)

    gain_run_folder = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/system_amplitude_calibration/season{args.season}/station{args.station}/results_per_run/"
    files = os.listdir(gain_run_folder)
    files_error = [f for f in files if f.endswith("error.csv")]
    files = [f for f in files if not f.endswith("error.csv")]

    combined_csv_path = f"absolute_amplitude_results/results_per_run/gains_per_run_season{args.season}_st{args.station}.csv"

    if not os.path.exists(combined_csv_path):
        runs = []
        dfs = []
        dfs_error = []
        for i, file in enumerate(files):
            run = int(file.split("run")[1].split(".csv")[0])
            runs.append(run)
            df = pd.read_csv(gain_run_folder + file)
            df.rename(columns={"Unnamed: 0" : "channel"}, inplace=True)
            df.insert(0, "run", run)
            dfs.append(df)

            df_error = pd.read_csv(gain_run_folder + files_error[i])
            df_error.rename(columns={"Unnamed: 0" : "channel"}, inplace=True)
            df_error.insert(0, "run", run)
            dfs_error.append(df_error)

        df = pd.concat(dfs, axis=0,ignore_index=True) 
        df_error = pd.concat(dfs_error, axis=0,ignore_index=True) 
        df.to_csv(combined_csv_path, index=False)
        df_error.to_csv(combined_csv_path.split(".csv")[0] + "_error" + ".csv", index=False)
    else:
        df = pd.read_csv(combined_csv_path)
        df_error = pd.read_csv(combined_csv_path.split(".csv")[0] + "_error" + ".csv")
    print(df["run"].unique())

    pdf = PdfPages(f"figures/variability/gain_run_distribution_season{args.season}_st{args.station}.pdf")
    pdf = PdfPages(f"figures/variability/gain_run_distribution_overview_season{args.season}_st{args.station}.pdf")

    for channel_id in channel_ids:
        confidence=0.95
        gains_errors = convert_error_to_db(df_error[df_error["channel"] == channel_id]["gain"].to_numpy(), df[df["channel"] == channel_id]["gain"].to_numpy())
        mean_gain_error = np.mean(gains_errors)
        gains = convert_to_db(df[df["channel"] == channel_id]["gain"].to_numpy())
        mean, sigma = np.mean(gains), np.std(gains, ddof=1)
        confidence_interval = stats.norm.interval(confidence, loc=mean, scale=sigma)
        np.set_printoptions(legacy="1.25")
        print(confidence_interval)

        plt.style.use("retro")
        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(gains, rwidth=0.9, bins=20)
    #    ax.text(0.7, 0.7, f"{100*confidence:.0f}% confidence interval:\n[{confidence_interval[0]:.2f},{confidence_interval[1]:.2f} dB]", transform=ax.transAxes)
        ax.vlines(confidence_interval, 0, max(hist), color="gray", ls="dashed", label = f"95% confidence interval:\n[{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
        ax.hlines(max(hist)/4, np.mean(gains) - mean_gain_error/2, np.mean(gains) + mean_gain_error/2, color="red", lw=2., label = f"mean error")

        ax.legend()
        ax.set_xlabel("Gain / dB")
        ax.set_ylabel("Counts")
        ax.set_title(f"Gain over run distribution season {args.season} station {args.station} channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()

    
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
