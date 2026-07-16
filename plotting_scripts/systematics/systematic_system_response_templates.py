import argparse
import copy
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


from utilities.utility_functions import convert_to_db



def calculate_energy_difference(frequencies, data, simulation,
                                freq_range=[0.15, 0.6]):
    df = np.diff(frequencies)[0]
    data = np.array(data)
    simulation = np.array(simulation)
    return np.sum(np.abs(data[(freq_range[0] < frequencies) & (frequencies < freq_range[1])] - simulation[(freq_range[0] < frequencies) & (frequencies < freq_range[1])])**2*df)



def integral_difference(frequencies, data, sim, freq_range):
    freq_mask = (freq_range[0] < frequencies) & (frequencies < freq_range[1])
    return np.trapezoid(np.abs(data[freq_mask] - sim[freq_mask]), frequencies[freq_mask])/np.trapezoid(np.abs(data[freq_mask]), frequencies[freq_mask])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", nargs="+", type=int)
    parser.add_argument("--use_relative_difference", action="store_true")
    parser.add_argument("--gain_in_dB", action="store_true")
    args = parser.parse_args()
    season = 2023
    station_ids = args.station

    
    use_relative_difference = args.use_relative_difference
    gain_in_dB = args.gain_in_dB

   
    fit_evaluation_range = [0.1, 0.7]


    channel_groups = {
            "VPols" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
            "HPols" : [4, 8, 11, 21],
            "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    channel_groups_stations = []
    for station_id in station_ids:
        channel_groups_tmp = copy.deepcopy(channel_groups)
        for key, ch_list in channel_groups.items():
            for ch_id in ch_list:
                if ch_id in known_broken_channels[str(season)][str(station_id)]:
                    channel_groups_tmp[key].remove(ch_id)
        channel_groups_stations.append(channel_groups_tmp)


    deep_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
    surface_channels = [12, 13, 14, 15, 16, 17, 18, 19, 20]

    system_response_paths = ["sim/library/system_response_templates_deep.json",
                             "sim/library/system_response_templates_surface.json"]

    with open(system_response_paths[0], "r") as f:
        deep_keys = list(json.load(f).keys())
    deep_keys.remove("time")

    with open(system_response_paths[1], "r") as f:
        surface_keys = list(json.load(f).keys())
    surface_keys += ["surface_query"]
    surface_keys.remove("time")
    # v3_ch5 was a test and does not contain a physical template
    surface_keys.remove("v3_ch5")


    data = []
    for station_id in station_ids:
        data_st = {}

        plot_data_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_plot_data.pickle"
        with open(plot_data_path, "rb") as plot_data_file:
            plot_data = pickle.load(plot_data_file)

        data_st["best_fit"] = {}
        data_st["best_fit"]["frequencies"] = plot_data["frequencies"]
        data_st["best_fit"]["data"] = plot_data["data"]
        data_st["best_fit"]["sim"] = plot_data["sim"]


        plot_data_tmpl_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_plot_data_all_templates.pickle"
        with open(plot_data_tmpl_path, "rb") as plot_data_file:
            plot_data_tmpl = pickle.load(plot_data_file)

        for key in plot_data_tmpl["sim"].keys():
            data_st[key] = {}
            data_st[key]["sim"] = plot_data_tmpl["sim"][key]
        data.append(data_st)



    calibration = []
    for station_id in station_ids:
        calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/default"
        calibration_settings_path = os.path.join(calibration_dir, "fit_settings.json")
        with open(calibration_settings_path, "r") as file:
            calibration_settings = json.load(file)

        template_keys = calibration_settings["response_templates_used"]
        calibration_st = {}
        for key in template_keys:
            calibration_file = os.path.join(calibration_dir,
                                            f"absolute_amplitude_calibration_season{season}_st{station_id}_key{key}.csv")
            calibration_st[key] = pd.read_csv(calibration_file)

        calibration_file = os.path.join(calibration_dir,
                                        f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")
        calibration_st["best_fit"] = pd.read_csv(calibration_file)
        calibration.append(calibration_st)


    deviation_from_best_fit = []
    gof = []
    gof_best_fit = []
    for antenna_type in channel_groups.keys():
        deviation_antenna_type = []
        gof_antenna_type = []
        gof_best_fit_antenna_type = []
        for station_idx, station_id in enumerate(station_ids):
            for channel_id in channel_groups_stations[station_idx][antenna_type]:
                gain_best_fit = calibration[station_idx]["best_fit"]["gain"][channel_id]
                if gain_in_dB:
                    gain_best_fit = convert_to_db(gain_best_fit)
                for key in template_keys:
                    if key == calibration[station_idx]["best_fit"]["best_fit_template"][channel_id]:
                        gof_tmp = integral_difference(data[station_idx]["best_fit"]["frequencies"],
                                                      data[station_idx]["best_fit"]["data"][channel_id],
                                                      data[station_idx][key]["sim"][channel_id],
                                                      fit_evaluation_range
                                                      )
                        gof_best_fit_antenna_type.append(gof_tmp)
                        continue
                    if channel_id in deep_channels and key in surface_keys:
                        continue
                    if channel_id in surface_channels and key in deep_keys:
                        continue


                    gof_tmp = integral_difference(data[station_idx]["best_fit"]["frequencies"],
                                                  data[station_idx]["best_fit"]["data"][channel_id],
                                                  data[station_idx][key]["sim"][channel_id],
                                                  fit_evaluation_range
                                                  )
                    gof_antenna_type.append(gof_tmp)


                    
                    gain_tmpl = calibration[station_idx][key]["gain"][channel_id]
                    if gain_in_dB:
                        gain_tmpl = convert_to_db(gain_tmpl)

                    deviation_tmp = np.abs(gain_best_fit - gain_tmpl)
                    if use_relative_difference:
                        deviation_tmp = (deviation_tmp/gain_best_fit) * 100
                    deviation_antenna_type.append(deviation_tmp)
        gof.append(gof_antenna_type)
        gof_best_fit.append(gof_best_fit_antenna_type)
        deviation_from_best_fit.append(deviation_antenna_type)
                                          
            





    plt.style.use("astroparticle_physics")
    pdf_path = f"figures/fitting_tests/effect_system_response_template_season{season}_all_stations_systematics.pdf"
    pdf = PdfPages(pdf_path)

    for i, antenna_type in enumerate(channel_groups.keys()):
        fig, ax = plt.subplots()
        color = (0.8, 0.2, 0.2)
        ax.hist(np.abs(deviation_from_best_fit[i]),
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.)
        xlabel = "Deviation from best fit G"
        if use_relative_difference:
            xlabel += " /%"
        elif gain_in_dB:
            xlabel += " /dB"
        else:
            xlabel += " /amplitude"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        fig.suptitle(antenna_type)

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()



    pdf_path = f"figures/fitting_tests/effect_system_response_template_season{season}_all_stations_systematics_gof.pdf"
    pdf = PdfPages(pdf_path)

    for i, antenna_type in enumerate(channel_groups.keys()):
        fig, ax = plt.subplots()
        color = (0.8, 0.2, 0.2)
        ax.hist(gof[i],
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.,
                label="all keys")
        color = (0.2, 0.2, 0.8)
        ax.hist(gof_best_fit[i],
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.,
                label="best_fit")
        xlabel = "goodness of fit (integrated difference)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.legend()
        fig.suptitle(antenna_type)

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()



    # TESTING FOR ONLY SPECIFIC CHANNELS

    channel_groups = {
            "PA" : [0, 1, 2, 3],
            "helper" : [5, 6, 7, 9, 10, 22, 23],
            "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
            }

    antenna_types_test = {
            "PA" : ["ch2", "ch2_6dB"],
            "helper" : ["ch9_6dB", "v2_ch11"],
            "LPDA" :  ["surface_query", "v2_ch13"]
            }


    channel_groups_stations = []
    for station_id in station_ids:
        channel_groups_tmp = copy.deepcopy(channel_groups)
        for key, ch_list in channel_groups.items():
            for ch_id in ch_list:
                if ch_id in known_broken_channels[str(season)][str(station_id)]:
                    channel_groups_tmp[key].remove(ch_id)
        channel_groups_stations.append(channel_groups_tmp)



    deviation_from_best_fit = []
    gof = []
    gof_best_fit = []
    for antenna_type in channel_groups.keys():
        deviation_antenna_type = []
        gof_antenna_type = []
        gof_best_fit_antenna_type = []
        for station_idx, station_id in enumerate(station_ids):
            for channel_id in channel_groups_stations[station_idx][antenna_type]:
                gain_best_fit = calibration[station_idx]["best_fit"]["gain"][channel_id]
                if gain_in_dB:
                    gain_best_fit = convert_to_db(gain_best_fit)
                for key in antenna_types_test[antenna_type]:
                    if key == calibration[station_idx]["best_fit"]["best_fit_template"][channel_id]:
                        gof_tmp = integral_difference(data[station_idx]["best_fit"]["frequencies"],
                                                      data[station_idx]["best_fit"]["data"][channel_id],
                                                      data[station_idx][key]["sim"][channel_id],
                                                      fit_evaluation_range
                                                      )
                        gof_best_fit_antenna_type.append(gof_tmp)
                        continue
                    if channel_id in deep_channels and key in surface_keys:
                        continue
                    if channel_id in surface_channels and key in deep_keys:
                        continue


                    gof_tmp = integral_difference(data[station_idx]["best_fit"]["frequencies"],
                                                  data[station_idx]["best_fit"]["data"][channel_id],
                                                  data[station_idx][key]["sim"][channel_id],
                                                  fit_evaluation_range
                                                  )
                    gof_antenna_type.append(gof_tmp)


                    
                    gain_tmpl = calibration[station_idx][key]["gain"][channel_id]
                    if gain_in_dB:
                        gain_tmpl = convert_to_db(gain_tmpl)

                    deviation_tmp = np.abs(gain_best_fit - gain_tmpl)
                    if use_relative_difference:
                        deviation_tmp = (deviation_tmp/gain_best_fit) * 100
                    deviation_antenna_type.append(deviation_tmp)
        gof.append(gof_antenna_type)
        gof_best_fit.append(gof_best_fit_antenna_type)
        deviation_from_best_fit.append(deviation_antenna_type)


    pdf_path = f"figures/fitting_tests/effect_system_response_template_season{season}_all_stations_systematics_limited_keys.pdf"
    pdf = PdfPages(pdf_path)

    for i, antenna_type in enumerate(channel_groups.keys()):
        fig, ax = plt.subplots()
        color = (0.8, 0.2, 0.2)
        ax.hist(np.abs(deviation_from_best_fit[i]),
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.)
        xlabel = "Deviation from best fit G"
        if use_relative_difference:
            xlabel += " /%"
        elif gain_in_dB:
            xlabel += " /dB"
        else:
            xlabel += " /amplitude"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        fig.suptitle(antenna_type)

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()



    pdf_path = f"figures/fitting_tests/effect_system_response_template_season{season}_all_stations_systematics_limited_keys_gof.pdf"
    pdf = PdfPages(pdf_path)

    for i, antenna_type in enumerate(channel_groups.keys()):
        fig, ax = plt.subplots()
        color = (0.8, 0.2, 0.2)
        ax.hist(gof[i],
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.,
                label="all keys")
        color = (0.2, 0.2, 0.8)
        ax.hist(gof_best_fit[i],
                histtype="stepfilled",
                facecolor=(*color, 0.5),
                edgecolor=(*color, 1.),
                lw=2.,
                label="best_fit")
        xlabel = "goodness of fit (integrated difference)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.legend()
        fig.suptitle(antenna_type)

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
