import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle, read_pickle, convert_to_db




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




def sim_with_weight(sim, calibration, channel_ids=np.arange(24)):
    response_paths = ["sim/library/deep_templates_combined.json","sim/library/v2_v3_surface_impulse_responses.json"]

    sim_spectra = []
    electronic_spectra = []
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
        galactic_spectrum = response["gain"](frequencies) * np.abs(sim["galactic"]["spectrum"][channel_id])

        ice_el_cross = sim["ice_el_cross"][channel_id]
        ice_gal_cross = sim["ice_gal_cross"][channel_id]
        el_gal_cross = sim["el_gal_cross"][channel_id]

        weight = calibration["el_ampl"][channel_id] * (frequencies - calibration["f0"][channel_id]) + 1

        sim_spectrum = calibration["gain"][channel_id] * np.sqrt(ice_spectrum**2 + weight**2 * electronic_spectrum**2 + galactic_spectrum**2 + weight * ice_el_cross + ice_gal_cross + weight * el_gal_cross)
        sim_spectra.append(sim_spectrum)
        electronic_spectra.append(calibration["gain"][channel_id] * weight * electronic_spectrum)

    return sim_spectra, electronic_spectra



def sim_no_weight(sim, calibration, channel_ids=np.arange(24)):
    response_paths = ["sim/library/deep_templates_combined.json","sim/library/v2_v3_surface_impulse_responses.json"]

    sim_spectra = []
    electronic_spectra = []
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
        galactic_spectrum = response["gain"](frequencies) * np.abs(sim["galactic"]["spectrum"][channel_id])

        ice_el_cross = sim["ice_el_cross"][channel_id]
        ice_gal_cross = sim["ice_gal_cross"][channel_id]
        el_gal_cross = sim["el_gal_cross"][channel_id]


        sim_spectrum = calibration["gain"][channel_id] * np.sqrt(ice_spectrum**2 + electronic_spectrum**2 + galactic_spectrum**2 + ice_el_cross + ice_gal_cross + el_gal_cross)
        sim_spectra.append(sim_spectrum)
        electronic_spectra.append(calibration["gain"][channel_id] * electronic_spectrum)

    return sim_spectra, electronic_spectra




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

    calibration_types = ["default", "measured_electronic_noise_no_weight"]


    calibration_paths = []
    for calibration_type in calibration_types:
        if calibration_type == "default":
            calibration_filename = calibration_name_base + "_best_fit.csv"
        else:
            calibration_filename = calibration_name_base + "_" + calibration_type + "_best_fit.csv"

        calibration_path = os.path.join(calibration_dir,"season" + str(season), "station" + str(station), calibration_type, calibration_filename)
        calibration_paths.append(calibration_path)

    calibration = {calibration_type : pd.read_csv(calibration_path) for calibration_type, calibration_path in zip(calibration_types, calibration_paths)}
    



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
    sim_dir_measured_electronic_noise = os.path.join(sim_dir_base, 
                                                     "complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise",
                                                     f"digitizer_v{digitizer_version}"
                                                     )

    sim_measured_electronic_noise = load_sim_components(sim_dir_measured_electronic_noise)
    

    if args.debug:
        print("DONE PARSING SIM")


    # we have two combinations (measured, weight), (measured, no weight)

    sim_types = {}
        


    sim_measured_electronic_noise_weight, sim_measured_electronic_noise_weight_electronic_comp = sim_with_weight(
            sim_measured_electronic_noise,
            calibration["default"],
            channel_ids=channel_ids)


    sim_measured_electronic_noise_no_weight, sim_measured_electronic_noise_no_weight_electronic_comp = sim_no_weight(
            sim_measured_electronic_noise,
            calibration["measured_electronic_noise_no_weight"],
            channel_ids=channel_ids)


    

    
    plt.style.use("retro")
    pdf_name = f"figures/electronic_noise/compare_electronic_noise_weight_season{season}_station{station}.pdf"
    pdf = PdfPages(pdf_name)

    for i , channel_id in enumerate(channel_ids):
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        ax_final_fit = axs[0]
        ax_final_fit.set_title("total fit")
        ax_final_fit.plot(frequencies_data, spectra_data[channel_id],
                          label="data",
                          lw=4.)
        ax_final_fit.plot(frequencies_data,
                          sim_measured_electronic_noise_weight[i],
                          label=f"measured noise\nwith fitted weight\n{calibration['default']['best_fit_template'][channel_id]}\n{convert_to_db(calibration['default']['gain'][channel_id]):.2f}",
                          lw=2.
                          )
        ax_final_fit.plot(frequencies_data,
                          sim_measured_electronic_noise_no_weight[i],
                          label=f"measured noise\nno weight\n{calibration['measured_electronic_noise_no_weight']['best_fit_template'][channel_id]}\n{convert_to_db(calibration['measured_electronic_noise_no_weight']['gain'][channel_id]):.2f}",
                          lw=2.
                          )

        ax_electronic = axs[1]
        ax_electronic.set_title("electronic noise")
        # to replace data such that colors match in two subplots
        ax_electronic.plot([1.1, 1.2], [0., 0.])
        ax_electronic.plot(frequencies_data,
                           sim_measured_electronic_noise_weight_electronic_comp[i],
                label="measured noise\nwith fitted weight")
        ax_electronic.plot(frequencies_data,
                           sim_measured_electronic_noise_no_weight_electronic_comp[i],
                label="measured noise\nno weight")

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



    # plot difference in gains


    fig, ax = plt.subplots()
    ax.scatter(channel_ids, convert_to_db(np.array(calibration["default"]["gain"])), label="measured noise with weight")
    ax.scatter(channel_ids, convert_to_db(np.array(calibration["measured_electronic_noise_no_weight"]["gain"])), label="measured noise no weight")
    ax.set_xlabel("channel")
    ax.set_ylabel("gain / dB")
    ax.set_xticks(channel_ids)
    ax.legend()
    fig.savefig(f"figures/electronic_noise/electronic_weight_gain_comparison_season{season}_st{station}.png")
