import argparse
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int)
    args = parser.parse_args()
    season = 2023
    station_id = args.station


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




    plot_data_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_plot_data.pickle"
    with open(plot_data_path, "rb") as plot_data_file:
        plot_data = pickle.load(plot_data_file)

          
    channel_ids = plot_data["channel_ids"]
    frequencies = plot_data["frequencies"]
    data = plot_data["data"]


    calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/default"
    calibration_settings_path = os.path.join(calibration_dir, "fit_settings.json")
    with open(calibration_settings_path, "r") as file:
        calibration_settings = json.load(file)

    template_keys = calibration_settings["response_templates_used"]
    calibration = {}
    for key in template_keys:
        calibration_file = os.path.join(calibration_dir,
                                        f"absolute_amplitude_calibration_season{season}_st{station_id}_key{key}.csv")
        calibration[key] = pd.read_csv(calibration_file)

    calibration_file = os.path.join(calibration_dir,
                                    f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")
    calibration["best_fit"] = pd.read_csv(calibration_file)

    

    plot_data_tmpl_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_plot_data_all_templates.pickle"
    with open(plot_data_tmpl_path, "rb") as plot_data_file:
        plot_data_tmpl = pickle.load(plot_data_file)

    
    energy_difference_tmpl = {key : [] for key in template_keys}
    energy_difference_tmpl["best_fit"] = []
    for key in template_keys:
        for channel_id in channel_ids:
            dE = calculate_energy_difference(plot_data_tmpl["frequencies"],
                                             plot_data_tmpl["data"][key][channel_id],
                                             plot_data_tmpl["sim"][key][channel_id])
            energy_difference_tmpl[key].append(dE)

    for channel_id in channel_ids:
        dE = calculate_energy_difference(plot_data["frequencies"],
                                         plot_data["data"][channel_id],
                                         plot_data["sim"][channel_id])
        energy_difference_tmpl["best_fit"].append(dE)




    plt.style.use("retro")
    pdf_path = f"figures/fitting_tests/effect_system_response_template_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_path)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].scatter(channel_ids, convert_to_db(calibration["best_fit"]["gain"]),
                   label="best fit", zorder=100)

    for channel_id in channel_ids:
        for key in template_keys:
            if channel_id in deep_channels:
                if key in surface_keys:
                    continue
            if channel_id in surface_channels:
                if key in deep_keys:
                    continue

            axs[0].scatter(channel_id, convert_to_db(calibration[key]["gain"][channel_id]),
                           color="gray", alpha=0.8)
    axs[0].legend()
    axs[0].set_ylabel("Gain / dB")


    axs[1].plot(channel_ids, np.zeros_like(channel_ids))
    for channel_id in channel_ids:
        for key in template_keys:
            if channel_id in deep_channels:
                if key in surface_keys:
                    continue
            if channel_id in surface_channels:
                if key in deep_keys:
                    continue

            axs[1].scatter(channel_id, convert_to_db(calibration[key]["gain"][channel_id]) - convert_to_db(calibration["best_fit"]["gain"][channel_id]),
                           color="gray", alpha=0.8)
    axs[1].set_ylabel("dGain\n(template - best fit)")

    axs[2].scatter(channel_ids, energy_difference_tmpl["best_fit"],
                   label="best fit", zorder=100)
    for channel_id in channel_ids:
        for key in template_keys:
            if channel_id in deep_channels:
                if key in surface_keys:
                    continue
            if channel_id in surface_channels:
                if key in deep_keys:
                    continue

            axs[2].scatter(channel_id, energy_difference_tmpl[key][channel_id],
                           color="gray", alpha=0.8)


    axs[2].set_xlabel("channels")
    axs[2].set_ylabel("Area between\ndata-sim curves")
    axs[2].set_yscale("log")
    for ax in axs:
        ax.set_xticks(channel_ids)
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    plt.close(fig)

    pdf.close()



    for channel_id in channel_ids:
        for key in template_keys:
            if energy_difference_tmpl[key][channel_id] < energy_difference_tmpl["best_fit"][channel_id]:
                print("--------------")
                print(f"channel {channel_id}")
                print(f"key {key}")
                print("==============")
                



