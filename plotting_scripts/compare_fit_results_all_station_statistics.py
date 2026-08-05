"""
code uses first fit given as default to compare to
e.g. to compare same response templates
this code looks at all stations
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd














if __name__ == "__main__":

    seasons = [2023]
    station_ids = [11]
    calibration_names = ["pulser_removed_slope_fitted", "pulser_removed_slope_fitted_lower_f0"]
    channel_ids = list(np.arange(24))
    

    calibration = []



    gof = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}
    gains = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}


    for season in seasons:
        for station_id in station_ids:
            for cal_i, calibration_name in enumerate(calibration_names):
                if cal_i == 0:
                    try:
                        best_fit_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
                        path = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_name}"
                        # baseline
                        calibration_tmp = pd.read_csv(path, index_col=0)
                    except FileNotFoundError:
                        best_fit_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_name}_best_fit.csv"
                        path = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_name}"
                        # baseline
                        calibration_tmp = pd.read_csv(path, index_col=0)

                    baseline_templates = calibration_tmp["best_fit_template"]         
                    other_parameters = {param_name : {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}
                                        for param_name in calibration_tmp.keys()}

                    for channel_id in channel_ids:
                        gains[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gain"][channel_id] 
                        gof[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gof_value"][channel_id]
                        for param_name in calibration_tmp.keys():
                            if param_name in ["gain", "gof", "gof_value", "best_fit_template"]:
                                continue
                            other_parameters[param_name][calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp[key][channel_id]



                else:
                    for channel_id in channel_ids: 
                        best_fit_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_name}_key{baseline_templates[channel_id]}.csv"
                        path = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_name}"

                        # baseline
                        calibration_tmp = pd.read_csv(path, index_col=0)

                        gains[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gain"][channel_id] 
                        gof[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gof"][channel_id]
                        for param_name in calibration_tmp.keys():
                            if param_name in ["gain", "gof", "gof_value", "best_fit_template"]:
                                continue
                            other_parameters[param_name][calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp[key][channel_id]




    channel_types = {
            "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA up" : [13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20],
            }


    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for calibration_name in calibration_names:
            axs[ax_i].hist(np.ndarray.flatten(gof[calibration_name][:, :, channel_ids]), 
                           label=calibration_name, bins=20,
                           )


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        

    fig.savefig("figures/fit_comparisons/compare_fit_gof.png")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for calibration_name in calibration_names[1:]:
            dG = (gains[calibration_name][:, :, channel_ids] - gains[calibration_names[0]][:, :, channel_ids]) / gains[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dG), 
                           label=calibration_name, bins=20,
                           )

            axs[ax_i].set_xlabel("dG / %")


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        

    fig.savefig("figures/fit_comparisons/compare_fit_gain.png")



    for param_name in other_parameters.keys():
        fig, axs = plt.subplots(2, 2)
        axs = np.ndarray.flatten(axs)
        for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
            for calibration_name in calibration_names:
                axs[ax_i].hist(np.ndarray.flatten(other_parameters[param_name][calibration_name][:, :, channel_ids]), 
                            label=calibration_name, bins=20,
                            )


            if ax_i == len(axs)-1:
                handles, labels = axs[ax_i].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center")
            

        fig.savefig(f"figures/fit_comparisons/compare_fit_{param_name}.png")