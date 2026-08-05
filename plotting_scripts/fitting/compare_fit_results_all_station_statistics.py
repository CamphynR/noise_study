"""
code uses first fit given as default to compare to
e.g. to compare same response templates
this code looks at all stations
"""
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd














if __name__ == "__main__":

    seasons = ["2024_radiant_v2"]
    # seasons = [2022, 2023, "2024_radiant_v2"]
    # seasons = [2022]
    station_ids = [11, 13, 23, 24]
    # calibration_names = ["pulser_removed_slope_fitted_norm", "pulser_removed_slope_fitted_lower_f0_norm"]
    calibration_names = ["default", "pulser_removed_slope_fitted_norm"]
    # calibration_names = ["slope_higher_freq_range"]
    # calibration_names = ["default"]
    # calibration_names = ["pulser_removed_slope_fitted_norm",
    #                      "test_increase_slope_10percent", "test_decrease_slope_10percent",
    #                      "test_increase_slope_100percent", "test_decrease_slope_90percent",
    #                      ]
    # calibration_names = ["pulser_removed_slope_fitted_norm",
    #                      "test_fit_with_2023_slope",
                        #  ]
    channel_ids = list(np.arange(24))

    do_per_season = False
    

    calibration = []

    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    gof = {key : np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan) for key in calibration_names}
    gains = {key : np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan) for key in calibration_names}


    for season_i, season in enumerate(seasons):
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
                    if season_i == 0 and cal_i==0 and station_id==station_ids[0]:
                        other_parameters = {param_name : {key : np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan) for key in calibration_names}
                                            for param_name in calibration_tmp.keys()
                                            if param_name not in ["gain", "gof_method", "gof", "gof_value", "best_fit_template"]}

                    for channel_id in channel_ids:
                        if channel_id in known_broken_channels[str(season)][str(station_id)]:
                            continue
                        gains[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gain"][channel_id] 
                        gof[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gof_value"][channel_id]
                        for param_name in calibration_tmp.keys():
                            if param_name in ["gain", "gof_method", "gof", "gof_value", "best_fit_template"]:
                                continue
                            other_parameters[param_name][calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp[param_name][channel_id]



                else:
                    for channel_id in channel_ids: 
                        if channel_id in known_broken_channels[str(season)][str(station_id)]:
                            continue
                        best_fit_name = f"absolute_amplitude_calibration_season{season}_st{station_id}_{calibration_name}_key{baseline_templates[channel_id]}.csv"
                        path = f"absolute_amplitude_results/season{season}/station{station_id}/{calibration_name}/{best_fit_name}"

                        # baseline
                        calibration_tmp = pd.read_csv(path, index_col=0)

                        gains[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gain"][channel_id] 
                        gof[calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp["gof"][channel_id]
                        for param_name in calibration_tmp.keys():
                            if param_name in ["gain", "gof", "gof_value", "best_fit_template"]:
                                continue
                            other_parameters[param_name][calibration_name][seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = calibration_tmp[param_name][channel_id]




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
        for cal_i, calibration_name in enumerate(calibration_names):
            axs[ax_i].hist(np.ndarray.flatten(gof[calibration_name][:, :, channel_ids]), 
                           label=calibration_name,
                           histtype="stepfilled",
                            facecolor=colors[cal_i] + "88",
                            edgecolor=colors[cal_i]
                           )


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        
        axs[ax_i].set_title(channel_type)

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_gof.png", bbox_inches="tight")

    

    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names[1:]):
            dgof = (gof[calibration_name][:, :, channel_ids] - gof[calibration_names[0]][:, :, channel_ids]) / gof[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dgof), 
                           label=calibration_name,
                           histtype="stepfilled",
                            facecolor=colors[cal_i] + "88",
                            edgecolor=colors[cal_i]
                           )


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        
        axs[ax_i].set_xlabel("dgof / %")
        axs[ax_i].set_title(channel_type)

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_dgof.png", bbox_inches="tight")



    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names):
            axs[ax_i].hist(np.ndarray.flatten(gains[calibration_name][:, :, channel_ids]), 
                           label=calibration_name,
                           histtype="stepfilled",
                            facecolor=colors[cal_i] + "88",
                            edgecolor=colors[cal_i]
                           )

        axs[ax_i].set_xlabel("Gain / amp")
        axs[ax_i].set_title(channel_type)


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_gain.png", bbox_inches="tight")

    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names[1:]):
            dG = (gains[calibration_name][:, :, channel_ids] - gains[calibration_names[0]][:, :, channel_ids]) / gains[calibration_names[0]][:, :, channel_ids]
            axs[ax_i].hist(100 * np.ndarray.flatten(dG), 
                           label=calibration_name,
                           histtype="stepfilled",
                            facecolor=colors[cal_i] + "88",
                            edgecolor=colors[cal_i]
                           )

        axs[ax_i].set_xlabel("dG / %")
        axs[ax_i].set_title(channel_type)


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_dgain.png", bbox_inches="tight")



    for param_name in other_parameters.keys():
        fig, axs = plt.subplots(2, 2)
        axs = np.ndarray.flatten(axs)
        for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
            for cal_i, calibration_name in enumerate(calibration_names):
                # for ch_i, slope_ch in enumerate(other_parameters["slope"][calibration_name][0, :, channel_ids]):
                #     for st_i, slope_st in enumerate(slope_ch):
                #         if slope_st > 1:
                #             print(station_ids[st_i])
                #             print(channel_ids[ch_i])
                #             print("-------------")
                axs[ax_i].hist(np.ndarray.flatten(other_parameters[param_name][calibration_name][:, :, channel_ids]), 
                            label=calibration_name,
                            histtype="stepfilled",
                            facecolor=colors[cal_i] + "88",
                            edgecolor=colors[cal_i]
                            )


            if ax_i == len(axs)-1:
                handles, labels = axs[ax_i].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center")
            axs[ax_i].set_title(channel_type)
            

        fig.tight_layout()
        fig.savefig(f"figures/fit_comparisons/compare_fit_{param_name}.png")




    if not do_per_season:
        exit()




    # SAME BUT PER SEASON
    # only to look at one fitting method for now (len(calibration_names) == 1)
    # --------------------
    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names):
            for season_i, season in enumerate(seasons):
                axs[ax_i].hist(np.ndarray.flatten(gof[calibration_name][season_i, :, channel_ids]), 
                            label=str(season),
                            histtype="stepfilled",
                                facecolor=colors[season_i] + "88",
                                edgecolor=colors[season_i]
                            )


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        
        axs[ax_i].set_title(channel_type)

    fig.savefig("figures/fit_comparisons/compare_fit_gof.png")

    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names):
            for season_i, season in enumerate(seasons):
                axs[ax_i].hist(np.ndarray.flatten(gains[calibration_name][season_i, :, channel_ids]), 
                            label=str(season),
                                histtype="stepfilled",
                                facecolor=colors[season_i] + "88",
                                edgecolor=colors[season_i]
                                )

        axs[ax_i].set_xlabel("dG / %")
        axs[ax_i].set_title(channel_type)


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_gain.png", bbox_inches="tight")


    for param_name in other_parameters.keys():
        fig, axs = plt.subplots(2, 2)
        axs = np.ndarray.flatten(axs)
        for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
            for cal_i, calibration_name in enumerate(calibration_names):
                for season_i, season in enumerate(seasons):
                    axs[ax_i].hist(np.ndarray.flatten(other_parameters[param_name][calibration_name][season_i, :, channel_ids]), 
                            label=str(season),
                                histtype="stepfilled",
                                facecolor=colors[season_i] + "88",
                                edgecolor=colors[season_i]
                                )


            if ax_i == len(axs)-1:
                handles, labels = axs[ax_i].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center")
            axs[ax_i].set_title(channel_type)
            axs[ax_i].set_xlabel(param_name)
            

        fig.tight_layout()
        fig.savefig(f"figures/fit_comparisons/compare_fit_{param_name}.png")


    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for calibration_name in calibration_names:
            for season_i, season in enumerate(seasons[1:]):
                dG = (gains[calibration_name][seasons.index(season), :, channel_ids] - gains[calibration_name][0, :, channel_ids]) / gains[calibration_name][0, :, channel_ids]
                axs[ax_i].hist(100 * np.ndarray.flatten(dG), 
                            label=f"{season} vs {seasons[seasons.index(season)-1]}",
                            histtype="stepfilled",
                                facecolor=colors[season_i] + "88",
                                edgecolor=colors[season_i]
                            )

        axs[ax_i].set_xlabel("dG / %")
        axs[ax_i].set_title(channel_type)


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05))
        

    fig.tight_layout()
    fig.savefig("figures/fit_comparisons/compare_fit_dgain.png", bbox_inches="tight")



    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        for cal_i, calibration_name in enumerate(calibration_names):
            for season_i, season in enumerate(seasons[1:]):
                dslope = other_parameters["slope"][calibration_names[0]][seasons.index(season), :, channel_ids] - other_parameters["slope"][calibration_names[0]][seasons.index(season) - 1, :, channel_ids]
                # for ch_i, dslope_channel in enumerate(dslope):
                #     for st_i, dslope_st in enumerate(dslope_channel):
                #         # print(dslope_station)
                #         if 100*dslope_st < -40:
                #             print("--------")
                #             print(100*dslope_st)
                #             print(station_ids[st_i])
                #             print(channel_ids[ch_i])
                #             print("--------")
                axs[ax_i].hist(np.ndarray.flatten(dslope), 
                        label=f"{season} vs {seasons[seasons.index(season)-1]}",
                            histtype="stepfilled",
                            facecolor=colors[season_i] + "88",
                            edgecolor=colors[season_i]
                            )


        if ax_i == len(axs)-1:
            handles, labels = axs[ax_i].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
        axs[ax_i].set_title(channel_type)
        axs[ax_i].set_xlabel("dslope / absolute_diff")
        

    fig.tight_layout()
    fig.savefig(f"figures/fit_comparisons/compare_fit_dslope.png")