import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
import NuRadioReco.utilities.signal_processing as signal
from NuRadioReco.utilities import units

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
    response_paths = ["sim/library/system_response_templates_deep.json",
                      "sim/library/system_response_templates_surface.json"]

    sim_spectra = []
    ice_spectra = []
    electronic_spectra = []
    galactic_spectra = []
    response_template_names = []
    response_templates = []
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

#        ice_spectra.append(calibration["gain"][channel_id] * ice_spectrum)
#        electronic_spectra.append(calibration["gain"][channel_id] * electronic_spectrum)
#        galactic_spectra.append(calibration["gain"][channel_id] * galactic_spectrum)

        ice_spectra.append(calibration["gain"][channel_id] * np.abs(sim["ice"]["spectrum"][channel_id]))
        electronic_spectra.append(calibration["gain"][channel_id] * np.abs(sim["electronic"]["spectrum"][channel_id]))
        galactic_spectra.append(calibration["gain"][channel_id] * np.abs(sim["galactic"]["spectrum"][channel_id]))

        response_template_names.append(calibration["best_fit_template"][channel_id])
        response_templates.append(response["gain"](frequencies))

    components = {"ice" : ice_spectra,
                  "electronic" : electronic_spectra,
                  "galactic" : galactic_spectra}

    return sim_spectra, components, response_template_names, response_templates




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int)
    parser.add_argument("--include_components", action="store_true")
    parser.add_argument("--include_response_on_plot", action="store_true")
    parser.add_argument("--include_one_comp_noise", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    season = 2023
    if season < 2024:
        digitizer_version = 2

    station = args.station

    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz

    channel_ids  = np.arange(24)
    if args.debug:
        channel_ids = [0, 4, 12, 13]
    channel_ids = [0, 4, 7, 12, 13, 23]

    calibration_dir = "/user/rcamphyn/noise_study/absolute_amplitude_results"
    calibration_name_base = f"absolute_amplitude_calibration_season{season}_st{station}"

    calibration_type = "default"


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


    sim_result, \
    sim_components, \
    response_template_names, \
    response_templates = sim_no_weight(
                                    sim,
                                    calibration,
                                    channel_ids=channel_ids)

    


    if args.include_one_comp_noise:
        effective_temperature_dir = "/user/rcamphyn/noise_study/effective_temperatures"
        effective_temperature_basename = f"eff_temperatures_calibrated_response_season{season}_st{station}.json"
        with open(os.path.join(effective_temperature_dir, effective_temperature_basename), "r") as file:
            effective_temperatures = json.load(file)


        nr_it = 5000
        generic_noise_adder = channelGenericNoiseAdder()
        bandwidth = [0.1, 0.7]
        noise_from_eff_temp = []
        for i, channel_id in enumerate(channel_ids):
            eff_temp = effective_temperatures[str(channel_id)]
            vrms_from_eff_temp = signal.calculate_vrms_from_temperature(eff_temp, bandwidth=bandwidth)
            noise = []
            for _ in range(nr_it):
                noise_tmp = generic_noise_adder.bandlimited_noise(*bandwidth, nr_samples, sampling_rate, amplitude=vrms_from_eff_temp, type="rayleigh", time_domain=False)
                noise_tmp = noise_tmp * calibration["gain"][channel_id] * response_templates[i] 
                noise.append(np.abs(noise_tmp)**2)
            noise_from_eff_temp.append(np.sqrt(np.mean(noise, axis=0)))



    # helper bandpass to clean components (in sim real bandpass is included in the system response template)
    bandpass_helper = channelBandPassFilter()
    bandpass = bandpass_helper.get_filter(frequencies_data, 11, 0, 0,
                                          passband=[0.1, 0.7], filter_type="butter", order=10)
    bandpass = np.abs(bandpass)
    
    plt.style.use("astroparticle_physics")
    pdf_name = f"figures/calibration_with_components_season{season}_st{station}.pdf"
    pdf = PdfPages(pdf_name)

    for i , channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.plot(frequencies_data, spectra_data[channel_id],
                          label="data",
                )
        ax.plot(frequencies_data,
                sim_result[i],
                label=f"Final fit",
                color="blue"
                          )




        if args.include_components:
            ax.plot(frequencies_data,
                              bandpass * sim_components["ice"][i],
                              label=f"ice\n(scaled by gain)",
                    ls="dashed",
                    color="red"
                              )
            ax.plot(frequencies_data,
                              bandpass * sim_components["electronic"][i],
                              label=f"electronic\n(scaled by gain)",
                    ls="dashed",
                    color="green"
                              )
            ax.plot(frequencies_data,
                              bandpass * sim_components["galactic"][i],
                              label=f"galactic\n(scaled by gain)",
                    ls="dashed",
                    color="deeppink"
                              )


        if args.include_response_on_plot:
            response_normalized = response_templates[i] / np.max(response_templates[i]) * np.max(sim_result[i]) / 2.
            ax.plot(frequencies_data, 
                    response_normalized,
                    label="Response template\n(normalized)",
                    ls="dashdot",
                    color="black")
        
        if args.include_one_comp_noise:
            ax.plot(frequencies_data,
                    noise_from_eff_temp[i],
                    label="one component\nnoise",
                    ls="dashdot",
                    color="deeppink")



        ax.set_xlim(0, 1.)
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("amplitude / V")
        ax.legend(loc="upper right", fontsize=21)
        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
