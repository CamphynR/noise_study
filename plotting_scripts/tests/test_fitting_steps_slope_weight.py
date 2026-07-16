from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd








if __name__ == "__main__":

    seasons = [2023]
    station_ids = [11, 24]
    calibration_names = ["test_system_response_weight", "test_system_response_fit_three_step", "test_system_response_fit_three_step_end_on_gain"]
    channel_ids = list(np.arange(24))
    

    calibration = []




    gof = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}
    gains = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}
    slope = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}
    slope_error = {key : np.zeros((len(seasons), len(station_ids), len(channel_ids))) for key in calibration_names}


    for season in seasons:
        for station_id in station_ids:
            for calibration_name in calibration_names:
                for channel_id in channel_ids: 
                    best_fit_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_name}_best_fit.csv"
                    best_fit_error_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_name}_best_fiterror.csv"
                    path = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_name}"
                    path_error = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_error_name}"

                    # baseline
                    calibration_tmp = pd.read_csv(path, index_col=0)
                    error_tmp = pd.read_csv(path_error, index_col=0)

                    gains[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gain"][channel_id] 
                    gof[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gof_value"][channel_id]
                    slope[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["slope"][channel_id]
                    slope_error[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = error_tmp["slope"][channel_id]




    channel_types = {
            "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA up" : [13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20],
            }


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
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
        

    fig.savefig("figures/tests/compare_fit_gof_slope_fitting.png")

    
    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names[1:]):
            dslope = (gof[calibration_name][:, :, channel_ids] - gof[calibration_names[0]][:, :, channel_ids]) / gof[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dslope), 
                           label=calibration_name, bins=20,
                           facecolor=(0, 0, 0, 0),
                           edgecolor=colors[cal_i],
                           lw=3.
                           )

            axs[ax_i].set_xlabel("dgof / %")


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        
    fig.suptitle(f"compared to {calibration_names[0]}")

    fig.savefig("figures/tests/compare_fit_gof_relative_slope_fitting.png", bbox_inches="tight")



    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names[1:]):
            dG = (gains[calibration_name][:, :, channel_ids] - gains[calibration_names[0]][:, :, channel_ids]) / gains[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dG), 
                           label=calibration_name,
                           facecolor=(0, 0, 0, 0),
                           edgecolor=colors[cal_i],
                           lw=3.
                           )

            axs[ax_i].set_xlabel("dG / %")


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        

    fig.savefig("figures/tests/compare_fit_gain_slope_fitting.png", bbox_inches="tight")
    

    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names):
            print(channel_type)
            print(calibration_name)
            mean_error_slope = np.mean(slope_error[calibration_name][:, :, channel_ids])
            print(f"mean error on slope: {mean_error_slope}")
            axs[ax_i].hist(np.ndarray.flatten(slope[calibration_name][:, :, channel_ids]), 
                           label=calibration_name, bins=20,
                           facecolor=(0, 0, 0, 0),
                           edgecolor=colors[cal_i],
                           lw=3.
                           )

            axs[ax_i].set_xlabel("slope")


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1))
        

    fig.savefig("figures/tests/compare_fit_slope_slope_fitting.png", bbox_inches="tight")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names[1:]):
            dslope = (slope[calibration_name][:, :, channel_ids] - slope[calibration_names[0]][:, :, channel_ids]) / slope[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dslope), 
                           label=calibration_name, bins=20,
                           facecolor=(0, 0, 0, 0),
                           edgecolor=colors[cal_i],
                           lw=3.
                           )

            axs[ax_i].set_xlabel("dslope / %")


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        
    fig.suptitle(f"compared to {calibration_names[0]}")

    fig.savefig("figures/tests/compare_fit_slope_relative_slope_fitting.png", bbox_inches="tight")
