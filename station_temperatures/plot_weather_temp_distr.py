import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--station", "-s", type=int, default=11)
    args = parser.parse_args()

    temperature_file_path = f"station_temperatures/weather/season{args.season}/weatherdata_st{args.station}_{args.season}-01-01to{args.season}-12-31.csv"
    
    df = pd.read_csv(temperature_file_path)


    spring_end = Time(f"{args.season}-06-20")
    summer_end = Time(f"{args.season}-09-20")
    autumn_end = Time(f"{args.season}-12-20")

    spring_df = df[df["time [unix s]"] < spring_end.unix]
    summer_df = df[(df["time [unix s]"] > spring_end.unix) & (df["time [unix s]"] < summer_end.unix)]
    autumn_df = df[(df["time [unix s]"] > summer_end.unix) & (df["time [unix s]"] < autumn_end.unix)]

    plt.style.use("retro")
    fig, axs = plt.subplots(3,1, sharex=True)
    spring_df.hist(column="temperature [°C]", ax=axs[0], bins=100, density=True, label="spring")
    axs[0].set_title("spring")
    summer_df.hist(column="temperature [°C]", ax=axs[1], bins=100, density=True, label="summer")
    axs[1].set_title("summer")
    autumn_df.hist(column="temperature [°C]", ax=axs[2], bins=100, density=True, label="summer")
    axs[2].set_title("autumn")
    axs[2].set_xlabel("temperature / C")

    for ax in axs:
        ax.set_ylabel("probability density")


    fig.savefig(f"station_temperatures/plot_weather_distr_season{args.season}_st{args.station}")
