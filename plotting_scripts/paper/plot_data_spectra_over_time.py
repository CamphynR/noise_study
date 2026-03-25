import argparse
import datetime
import glob
import matplotlib.pyplot as plt
import natsort
import numpy as np


from utilities.utility_functions import read_freq_spectrum_from_pickle, rolling_average


def get_month(string):
    string = string.split("/")[-1]
    string = string.split("month_")[-1]
    month = string.split("_")[0]
    return int(month)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--station", type=int)
    args = parser.parse_args()




    bandpass = [0.1, 0.7]
    fit_range = [0.15, 0.6]

    season = args.season
    station_id = args.station
    
    channels = [0, 4, 13, 14]
    representative_channel_names = ["PA", "HPol", "LPDA up", "LPDA down"]

    spectra_files = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{season}/station{station_id}/clean/average_ft_month_*_combined.pickle")
    spectra_files = natsort.natsorted(spectra_files)

    months = []
    spectra = []

    for i, spectra_file in enumerate(spectra_files):
        spectra_dict = read_freq_spectrum_from_pickle(spectra_file)
        if i == 0:
            frequencies = spectra_dict["frequencies"]    
        spectra_tmp = spectra_dict["spectrum"]
        months.append(get_month(spectra_file))
        spectra.append(spectra_tmp)


    months_string = [datetime.datetime(season, month, 1).strftime("%B") for month in months]
    spectra = np.array(spectra)
    
    ratio = 100 * (spectra - spectra[0]) / spectra[0]
    ratio[:, :, frequencies < bandpass[0]] = 0
    ratio[:, :, frequencies > bandpass[1]] = 0


    
    plt.style.use("astroparticle_physics")
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    # re-arange for clarity on the plot
    colors = np.roll(colors, 2)
    fig, axs = plt.subplots(2, 2, figsize=(20,10), sharey=True)
    axs = np.ndarray.flatten(axs)
    for i, ax in enumerate(axs):
        for month in range(len(ratio)):
            if month == 0:
                # baseline
                zorder = 1000
            else:
                zorder = 0
            res=20
            ax.plot(frequencies[int(res/2):-int(res/2)], rolling_average(ratio[month][channels[i]],
                                                 resolution=res),
#            ax.plot(frequencies, ratio[month][channels[i]],
                    label=f"{months_string[month]}",
                    color=colors[month],
                    lw=1.5,
                    zorder=zorder)
        ax.axvspan(*fit_range, alpha=0.2, label = "fit range", edgecolor="black", linestyle="dashed")
        ax.set_xlim(0.1, 0.7)
        ax.legend(loc="lower right", ncols=2)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("ratio / %")
        ax.set_title(representative_channel_names[i])
    
    fig.tight_layout()
    fig.savefig(f"figures/paper/spectra_per_month_season{season}_st{station_id}.png", dpi=300, bbox_inches="tight")

