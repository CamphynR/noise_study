import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle, read_pickle




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




def sim_no_weight(sim, calibration, channel_ids=np.arange(24)):
    response_paths = ["sim/library/deep_templates_combined.json","sim/library/v2_v3_surface_impulse_responses.json"]

    sim_spectra = []
    ice_spectra = []
    electronic_spectra = []
    galactic_spectra = []
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

        ice_el_cross = response["gain"](frequencies) * sim["ice_el_cross"][channel_id]
        ice_gal_cross = response["gain"](frequencies) * sim["ice_gal_cross"][channel_id]
        el_gal_cross = response["gain"](frequencies) * sim["el_gal_cross"][channel_id]


        sim_spectrum = calibration["gain"][channel_id] * np.sqrt(ice_spectrum**2 + electronic_spectrum**2 + galactic_spectrum**2 + ice_el_cross + ice_gal_cross + el_gal_cross)
        sim_spectra.append(sim_spectrum)

        ice_spectra.append(calibration["gain"][channel_id] * ice_spectrum)
        electronic_spectra.append(calibration["gain"][channel_id] * electronic_spectrum)
        galactic_spectra.append(calibration["gain"][channel_id] * galactic_spectrum)

    components = {"ice" : ice_spectra,
                  "electronic" : electronic_spectra,
                  "galactic" : galactic_spectra}

    return sim_spectra, components




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    season = 2023
    if season < 2024:
        digitizer_version = 2

    station = 11

    channel_ids  = np.arange(24)
    if args.debug:
        channel_ids = [0, 4, 12, 13]

    calibration_dir = "/user/rcamphyn/noise_study/absolute_amplitude_results"
    calibration_name_base = f"absolute_amplitude_calibration_season{season}_st{station}"

    calibration_type = "measured_noise_no_weight_new_impedance_cable_11"


    if calibration_type == "default":
        calibration_filename = calibration_name_base + "_best_fit.csv"
    else:
        calibration_filename = calibration_name_base + "_" + calibration_type + "_best_fit.csv"

    calibration_path = os.path.join(calibration_dir,"season" + str(season), "station" + str(station), calibration_type, calibration_filename)

    calibration = pd.read_csv(calibration_path)



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
                                                     "complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise_new_impedance_mismatch",
                                                     f"digitizer_v{digitizer_version}"
                                                     )

    sim = load_sim_components(sim_dir)
    


    if args.debug:
        print("DONE PARSING SIM")


    sim_result, sim_components = sim_no_weight(
                                    sim,
                                    calibration,
                                    channel_ids=channel_ids)

    

    
    plt.style.use("astroparticle_physics")
    pdf_name = f"figures/calibration_with_components_season{season}_st{station}.pdf"
    pdf = PdfPages(pdf_name)

    for i , channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots(1, 1)
        ax.plot(frequencies_data, spectra_data[channel_id],
                          label="data",
                )
        ax.plot(frequencies_data,
                          sim_result[i],
                          label=f"Final fit",
                          )




        ax.plot(frequencies_data,
                          sim_components["ice"][i],
                          label=f"ice",
                ls="dashed",
                          )
        ax.plot(frequencies_data,
                          sim_components["electronic"][i],
                          label=f"electronic",
                ls="dashed",
                          )
        ax.plot(frequencies_data,
                          sim_components["galactic"][i],
                          label=f"galactic",
                ls="dashed",
                          )

        ax.set_xlim(0, 1.)
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("amplitude / V")
        ax.legend()
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
