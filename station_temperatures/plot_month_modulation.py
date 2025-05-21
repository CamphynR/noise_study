import argparse
from astropy.time import Time
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", nargs="+")
    args = parser.parse_args()
    
    station_id = os.path.basename(args.data[0]).split("st")[1][0:2]
       
    years = []
    times_per_year = []
    temps_per_year = []
    for d in args.data:
        year = d.split(f"st{station_id}_")[1].split("-09-01")[0]
        years.append(int(year))
        time = []
        temperature = []
        with open(d, "r") as temp_file:
            reader = csv.DictReader(temp_file)
            for i, row in enumerate(reader):
                time_tmp = Time(float(row["time [unix s]"]), format="unix")
                t_ref = Time(f"{year}-{time_tmp.datetime.month}-{time_tmp.datetime.day}T00:00:00", format="isot", scale="utc")
                time.append((time_tmp - t_ref).sec)
                try:
                    temperature.append(float(row["heat [\N{DEGREE SIGN}C]"]))
                except:
                    temperature.append(float(row["temperature [\N{DEGREE SIGN}C]"]))
        times_per_year.append(time)
        temps_per_year.append(temperature)
    
    plt.style.use("retro")
    for i, year in enumerate(years):
        plt.plot(times_per_year[i], temps_per_year[i], label=year)
#    time_date = [datetime.datetime.fromtimestamp(t).strftime("%d") for t in times_per_year[0]]
#    plt.xticks(times_per_year[0], labels=time_date, rotation=-45, ha="left")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("time / day")
    plt.ylabel("temperature / \N{DEGREE SIGN}C")
    plt.title(f"Station {station_id}")
    plt.savefig(f"station_temperatures/plot_monthly_mod_st{station_id}", dpi=200, bbox_inches="tight")
