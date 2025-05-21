import argparse
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d")
    args = parser.parse_args()
       
    time = []
    temperature = []
    with open(args.data, "r") as temp_file:
        reader = csv.DictReader(temp_file)
        count = -1
        time_tmp = []
        temperature_tmp = []
        for i, row in enumerate(reader):
            count += 1
            time_tmp.append(float(row["time [unix s]"]))
            try:
                temperature_tmp.append(float(row["heat [\N{DEGREE SIGN}C]"]))
            except:
                temperature_tmp.append(float(row["temperature [\N{DEGREE SIGN}C]"]))
            if count == 1000:
                count = 0
                time.append(np.median(time_tmp))
                temperature.append(np.mean(temperature_tmp))
                time_tmp = []
                temperature_tmp = []
    
    time_date = [datetime.datetime.fromtimestamp(t).strftime("%Y-%b-%d") for t in time]

    station_id = os.path.basename(args.data).split("st")[1][0:2]

    plt.style.use("gaudi")
    plt.plot(time, temperature)
    plt.xticks(time[::30], labels=time_date[::30], rotation=-45, ha="left")
    base = os.path.basename(args.data).split(".")[0]
    plt.tight_layout()
    plt.xlabel("time")
    plt.ylabel("temperature / \N{DEGREE SIGN}C")
    plt.title(f"Station {station_id}")
    plt.savefig(f"station_temperatures/plot_{base}", dpi=200, bbox_inches="tight")
