import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


from utilities.utility_functions import convert_to_db





if __name__ == "__main__":
    
    season = 2023
    station_id = 11

    fit_range = [0.15, 0.6]

    calibration_dir = f"absolute_amplitude_results/season{season}/station{station_id}/default"
    calibration_plot = os.path.join(calibration_dir, f"absolute_amplitude_calibration_season{season}_st{station_id}_plot_data.pickle")
    calibration_results = os.path.join(calibration_dir, f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")



    with open(calibration_plot, "rb") as file:
        plot_data = pickle.load(file)

    calibration = pd.read_csv(calibration_results, index_col=0)
    print(calibration)


    # should be for 2x2 plot
    representative_channels = {"Phased Array" : 0,
                               "HPol" : 4,
                               "LPDA up" : 13,
                               "LPDA down" : 12}




    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 6))
    axs = np.ndarray.flatten(axs)

    
    frequencies = plot_data["frequencies"]
    for i, (antenna_type, channel_id) in enumerate(representative_channels.items()):
        data_spectrum = plot_data["data"][channel_id]
        sim_spectrum = plot_data["sim"][channel_id]

        axs[i].plot(frequencies, data_spectrum, label="data")
        axs[i].plot(frequencies, sim_spectrum, label="simulation")
        axs[i].text(0.95, 0.95, f"G={convert_to_db(calibration['gain'][channel_id]):.2f} dB",
                    va="top",
                    ha="right",
                    transform=axs[i].transAxes,
                    fontsize=12,
                    bbox={"facecolor" : "white",
                          "boxstyle" : "round"})
        if i==0:
            axs[i].legend(loc="lower right")

        axs[i].set_title(antenna_type)    

            

    for ax in axs:
        ax.set_xlim(0., 1.)
        ax.axvspan(0, fit_range[0],
                    alpha=0.2,
                   edgecolor="black",
                   color="gray",
                   linestyle="dashed")
        ax.axvspan(fit_range[1], 1.6,
                   alpha=0.2,
                   edgecolor="black",
                   color="gray",
                   linestyle="dashed")


        
            

    fig.text(0.5, 0.01, 'frequencies / GHz', ha='center')
    fig.text(0.005, 0.5, 'amplitude / V', va='center', rotation='vertical')

    fig.tight_layout()
    fig.savefig(f"figures/paper/representative_results_season{season}_st{station_id}")   


