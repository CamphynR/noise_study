import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle, read_pickle, convert_to_db



def read_json(path):
    with open(path, "r") as file:
        content = json.load(file)
    return content



def load_sim_components(sim_dir):
    components = ["ice", "electronic" , "galactic"]
    sim_paths = [os.path.join(sim_dir,
                              component,
                              f"station{station}",
                              "clean",
                              "average_ft.pickle"
                              ) for component in components]

#    sim_dict = {component : read_freq_spectrum_from_pickle(component_path)
#                for component, component_path in zip(components, sim_paths)}
    sim_dict = {}
    for component, component_path in zip(components, sim_paths):
        if args.debug:
            print(f"reading {component_path}")
        sim_dict[component] = read_freq_spectrum_from_pickle(component_path)
    sim_cross_path = os.path.join(sim_dir,
                                  "cross_products",
                                  f"station{station}",
                                  "cross_products.pickle")

    if args.debug:
        print("reading sim_cross_path")
    sim_cross = read_pickle(sim_cross_path)
    
    for key in ["ice_el_cross", "ice_gal_cross", "el_gal_cross"]:
        sim_dict[key] = sim_cross[key]
    return sim_dict




def calculate_sim(sim, calibration, fit_settings, channel_ids=np.arange(24)):
    response_paths = ["sim/library/system_response_templates_deep.json","sim/library/system_response_templates_surface.json"]


    sim_spectra = []
    electronic_spectra = []
    galactic_spectra = []
    galactic_spectra_raw = []
    for channel_id in channel_ids:
        response_helper = systemResponseTimeDomainIncorporator()
        response_helper.begin(response_path=response_paths,
                              overwrite_key=calibration["best_fit_template"][channel_id],
                              bandpass_kwargs=dict(passband=[0.1, 0.7],
                                                   filter_type="butter",
                                                   order=10))
        response = response_helper.get_response(channel_id=channel_id)

        frequencies = sim["ice"]["frequencies"]


        ice_spectrum = response["gain"](frequencies) * np.abs(sim["ice"]["spectrum"][channel_id])
        electronic_spectrum = response["gain"](frequencies) * np.abs(sim["electronic"]["spectrum"][channel_id])
        galactic_spectrum_raw = np.abs(sim["galactic"]["spectrum"][channel_id])
        galactic_spectrum = response["gain"](frequencies) * np.abs(sim["galactic"]["spectrum"][channel_id])

        if "scale_noise_components" in fit_settings.keys():
            if fit_settings["scale_noise_components"] is not None:
                print("testing found comp scale")
                galactic_spectrum *= fit_settings["scale_noise_components"]["galactic"][channel_id]
                galactic_spectrum_raw *= fit_settings["scale_noise_components"]["galactic"][channel_id]

        ice_el_cross = sim["ice_el_cross"][channel_id]
        ice_gal_cross = sim["ice_gal_cross"][channel_id]
        el_gal_cross = sim["el_gal_cross"][channel_id]


        sim_spectrum = calibration["gain"][channel_id] * np.sqrt(ice_spectrum**2 + electronic_spectrum**2 + galactic_spectrum**2 + ice_el_cross + ice_gal_cross + el_gal_cross)
        sim_spectra.append(sim_spectrum)
        electronic_spectra.append(calibration["gain"][channel_id] * electronic_spectrum)
        galactic_spectra.append(calibration["gain"][channel_id] * galactic_spectrum)
        galactic_spectra_raw.append(galactic_spectrum_raw)

    return sim_spectra, galactic_spectra_raw




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, default=11)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    season = 2023
    if season < 2024:
        digitizer_version = 2
    station = args.station
    channel_ids  = np.arange(24)
    if args.debug:
        channel_ids = [0, 4, 12, 13]

    calibration_dir = "/user/rcamphyn/noise_study/absolute_amplitude_results"
    calibration_name_base = f"absolute_amplitude_calibration_season{season}_st{station}"

    calibration_types = ["default", "gal_gsm2016_down_2.5percent", "gal_lfss_up_7percent"]


    calibration_paths = []
    fit_settings_path = []
    for calibration_type in calibration_types:
        if calibration_type == "default":
            calibration_filename = calibration_name_base + "_best_fit.csv"
        else:
            calibration_filename = calibration_name_base + "_" + calibration_type + "_best_fit.csv"

        calibration_path = os.path.join(calibration_dir,"season" + str(season), "station" + str(station), calibration_type, calibration_filename)
        calibration_paths.append(calibration_path)

        fit_setting_path_tmp = os.path.join(calibration_dir,"season" + str(season), "station" + str(station), calibration_type, "fit_settings.json")
        fit_settings_path.append(fit_setting_path_tmp)

    calibration = {calibration_type : pd.read_csv(calibration_path) for calibration_type, calibration_path in zip(calibration_types, calibration_paths)}
    fit_settings = {calibration_type : read_json(fit_setting_path) for calibration_type, fit_setting_path in zip(calibration_types, fit_settings_path)}
    



    # DATA
    if args.debug:
        print("PARSING DATA")

    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{season}/station{station}/clean/average_ft_combined.pickle"
    data = read_freq_spectrum_from_pickle(data_path)
    frequencies_data = data["frequencies"]
    spectra_data = data["spectrum"]

    if args.debug:
        print("DONE PARSING DATA")

    # SIM
    if args.debug:
        print("PARSING SIM")

    sim_dir_base = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft"


    # with measured electronic noise
    sim_dir = os.path.join(sim_dir_base, 
                                         "default",
                                         f"digitizer_v{digitizer_version}"
                                         )

    sim_comp = load_sim_components(sim_dir)
    

    if args.debug:
        print("DONE PARSING SIM")



        

    sim_default, sim_default_galaxy = calculate_sim(
            sim_comp,
            calibration["default"],
            fit_settings["default"],
            channel_ids=channel_ids)


    sim_gsm2016, sim_gsm2016_galaxy = calculate_sim(
            sim_comp,
            calibration["gal_gsm2016_down_2.5percent"],
            fit_settings["gal_gsm2016_down_2.5percent"],
            channel_ids=channel_ids)



    sim_lfss, sim_lfss_galaxy = calculate_sim(
            sim_comp,
            calibration["gal_lfss_up_7percent"],
            fit_settings["gal_lfss_up_7percent"],
            channel_ids=channel_ids)
    



    plt.style.use("astroparticle_physics")
    pdf_name = f"figures/galactic_noise/compare_galaxy_noise_season{season}_station{station}.pdf"
    pdf = PdfPages(pdf_name)

    for i , channel_id in enumerate(channel_ids):
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        ax_final_fit = axs[0]
        ax_final_fit.set_title("total fit")
        ax_final_fit.plot(frequencies_data, spectra_data[channel_id],
                          label="data",
                          lw=4.)
        ax_final_fit.plot(frequencies_data,
                          sim_default[i],
                          label=f"default",
                          lw=2.
                          )
        ax_final_fit.plot(frequencies_data,
                          sim_gsm2016[i],
                          label="GSM2016",
                          lw=2.)
        ax_final_fit.plot(frequencies_data,
                          sim_lfss[i],
                          label="LFSM",
                          lw=2.)

        ax_galactic = axs[1]
        ax_galactic.set_title("galactic contribution (before response)")
        # to replace data such that colors match in two subplots
        ax_galactic.plot([1.1, 1.2], [0., 0.])
        ax_galactic.plot(frequencies_data,
                           sim_default_galaxy[i],
                label="default")
        ax_galactic.plot(frequencies_data,
                           sim_gsm2016_galaxy[i],
                label="GSM2016")
        ax_galactic.plot(frequencies_data,
                           sim_lfss_galaxy[i],
                label="LFSM")

        for ax in axs:
            ax.set_xlim(0, 1.)
            ax.set_xlabel("frequency / GHz")
            ax.set_ylabel("amplitude / V")
            ax.legend()
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()


