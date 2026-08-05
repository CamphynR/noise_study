from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


def compare_spectra(season, station_id, default_calibration):
    default_calibration_path=f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_plot_data.pickle"
    electronic_variation_path=f"absolute_amplitude_results/season{season}/station{station_id}/electronic_temperature_varied/absolute_amplitude_calibration_season{season}_st{station_id}_electronic_temperature_varied_plot_data_all_templates.pickle"

    with open(default_calibration_path, "rb") as file:
        default_calibration_plot_data = pickle.load(file)

    
    with open(electronic_variation_path, "rb") as file:
        electronic_variation_plot_data = pickle.load(file)


    default_keys = default_calibration["best_fit_template"]

    channel_ids = electronic_variation_plot_data["channel_ids"]

    figname = f"figures/electronic_temp_spectra_season{season}_st{station_id}.pdf"
    pdf = PdfPages(figname)
    for channel_id in channel_ids:

        electronic_variation_sim = electronic_variation_plot_data["sim"][default_keys[channel_id]][channel_id]
    
        fig, ax = plt.subplots()
        ax.plot(default_calibration_plot_data["frequencies"],
                default_calibration_plot_data["data"][channel_id],
                lw=2.,
                label="data")

        ax.plot(default_calibration_plot_data["frequencies"],
                default_calibration_plot_data["sim"][channel_id],
                lw=2.,
                label="default calibration")


        ax.plot(electronic_variation_plot_data["frequencies"],
                electronic_variation_sim,
                lw=2.,
                label="electronic temperature 20C")


        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("amplitude / V/GHz")
        ax.legend()

        ax.set_xlim(0., 1.)
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)


    pdf.close()




if __name__ == "__main__":
    plot_spectra = True
    seasons = ["2022", "2023"]
    station_ids = [11, 12, 13, 21, 22, 23, 24]
    channel_ids = np.arange(24)

    gain_temp_default = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    gain_temp_20C = np.zeros((len(seasons), len(station_ids), len(channel_ids)))
    for season_idx, season in enumerate(seasons):
        for station_idx, station_id in enumerate(station_ids):
#            if season == "2024_radiant_v2" and station_id in [21, 22]:
#                continue
#            if season == "2022" and station_id in [22]:
#                continue
            if season == "2023" and station_id in [22]:
                continue

            temp_default_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default"
            temp_20C_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/electronic_temperature_varied"

            temp_default_calibration = pd.read_csv(
                    os.path.join(temp_default_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"),
                    index_col=0)
            for channel_idx, channel_id in enumerate(channel_ids):
                gain_temp_default[season_idx, station_idx, channel_idx] = temp_default_calibration["gain"][channel_id]
                best_fit_template_default_temp = temp_default_calibration["best_fit_template"][channel_id]
                temp_20C_calibration = pd.read_csv(
                        os.path.join(temp_20C_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_electronic_temperature_varied_key{best_fit_template_default_temp}.csv")
                        )
                gain_temp_20C[season_idx, station_idx, channel_idx] = temp_20C_calibration["gain"][channel_id]



    dG = (gain_temp_default - gain_temp_20C) / gain_temp_default


    channel_groups = {"Vpol (-40C -> 20C)" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                      "Hpol (-40C -> 20C)" : [4, 8, 11, 21],
                      "LPDA up (0C -> 20C) " : [13, 16, 19],
                      "LPDA down (0C -> 20C)" : [12, 14, 15, 17, 18, 20]}

    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(2, 2, figsize = (12, 8))
    axs = np.ndarray.flatten(axs)

    for ax_i, (channel_type, channel_ids) in enumerate(channel_groups.items()):
        axs[ax_i].hist(np.ndarray.flatten(dG[:, :, channel_ids]) * 100,
                edgecolor=colors[ax_i],
                facecolor=colors[ax_i] + "88",
                lw = 3.,
                histtype="stepfilled",
                       label=channel_type)

    #    ax.hist(np.ndarray.flatten(gof_hpol_v4),
    #            edgecolor=colors[1],
    #            facecolor=colors[1] + "88",
    #            lw = 2.5,
    #            label= "hpol v4",
    #            histtype="stepfilled")

    for ax in axs:
        ax.set_xlabel("dG / %")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/electronic_temp")


    for season_idx, season in enumerate(seasons):
        for station_idx, station_id in enumerate(station_ids):
            if season == "2023" and station_id in [22]:
                continue
            temp_default_calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default"
            temp_default_calibration = pd.read_csv(
                    os.path.join(temp_default_calibration_path,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"),
                    index_col=0)
            if plot_spectra:
                compare_spectra(season, station_id, temp_default_calibration)
