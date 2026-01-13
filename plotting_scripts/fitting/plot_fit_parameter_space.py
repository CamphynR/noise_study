import argparse
import json
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats

from NuRadioReco.utilities import fft, units

from fitting.spectrumFitter import spectrumFitter
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle, convert_to_db



def calculate_reduced_chi2(spectrum_1, spectrum_2, var_1, frequencies, freq_range):
    mask = (frequencies > freq_range[0]) & (frequencies < freq_range[1])
    spectrum_1 = np.abs(spectrum_1[mask])
    spectrum_2 = np.abs(spectrum_2[mask])
    var_1 = var_1[mask]
    ndof = len(spectrum_1)

    chi2 = np.sum((spectrum_1 - spectrum_2)**2 / var_1**2)
    chi2_reduced = chi2/ndof

    return chi2_reduced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--station", "-s", default=12)
    args = parser.parse_args()

    # SETTINGS
    channel_mapping = {
        "deep" : [0, 1, 2, 3],
        "helper" : [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
        "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
        }

    season = args.season
    digitizer_version = "digitizer_v3" if season > 2023 \
                  else "digitizer_v2"
    station_id = args.station
    channel_ids = np.arange(24)
    sampling_rate = 2.4 * units.GHz if season > 2023 \
            else 3.2 * units.GHz

    goodness_of_fit_variable = "reduced_chi2"

    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)


    parameter_ranges = dict(gain = np.arange(300, 1500, 20),
                            el_ampl = np.arange(0, 10, 0.1),
                            el_cst = np.arange(0, 10, 1),
                            f0 = 150)

    gain_range = parameter_ranges["gain"]
    el_ampl_range = parameter_ranges["el_ampl"]
    el_cst = 1
    f0 = 0.15


    # DATA
    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{season}/station{station_id}/clean/average_ft_combined.pickle"
    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response/{digitizer_version}"
    sim_paths = [os.path.join(sim_dir, f"{component}/station{station_id}/clean/average_ft.pickle")
                 for component in ["ice", "electronic", "galactic"]]


    cross_products_path = f"{sim_dir}/cross_products/station{station_id}/cross_products.pickle"


    # SYSTEM RESPONSE TEMPLATE
    system_response_paths = ["sim/library/v2_v3_deep_impulse_responses_for_comparison.json",
                             "sim/library/v2_v3_surface_impulse_responses.json"]

    

    fit_results_templates = {}
    fit_functions = {}
    mode = "electronic_temp_cross"


    system_response = systemResponseTimeDomainIncorporator()
    system_response.begin(det=0, response_path=system_response_paths,
                          bandpass_kwargs=bandpass_kwargs, overwrite_key="v2_ch2")

    fitter = spectrumFitter(data_path,
                            sim_paths,
                            cross_products_path=cross_products_path,
                            sampling_rate=sampling_rate,
                            system_response=system_response,
                            include_impedance_mismatch_correction=True)


    fit_idxs = fitter.fit_idxs
    cost_function = fitter.cost_function
    
    
    plt.style.use("retro")
    for channel_id in channel_ids:
        parameter_space = np.zeros((len(gain_range), len(el_ampl_range)))
        fig, ax = plt.subplots()
        channel_fit_function = fitter.get_fit_function(mode, channel_id)
        x_data = fitter.frequencies[fit_idxs]
        y_data = fitter.data_spectrum[channel_id][fit_idxs]
        y_err = fitter.var_data_spectrum[channel_id][fit_idxs]
        for i, gain_step in enumerate(gain_range):
            for j, el_ampl_step in enumerate(el_ampl_range):
                cost_function_tmp = cost_function(x=x_data, y=y_data, yerror=y_err, model=channel_fit_function)
                cost = cost_function_tmp(gain_step, el_ampl_step, el_cst, f0)
                parameter_space[i][j] = np.abs(cost)
    
        argmin = np.where(parameter_space==np.min(parameter_space))
        cmesh = ax.pcolormesh(gain_range, el_ampl_range, np.log(parameter_space).T, shading="gouraud")
        cbar = fig.colorbar(cmesh)
        cbar.set_label("Log Chi Squared")
        ax.scatter(gain_range[argmin[0]], el_ampl_range[argmin[1]], label = f"min at\ngain: {gain_range[argmin[0]]}\nel_ampl:{el_ampl_range[argmin[1]]}", color="red")
        ax.legend()
        ax.set_xlabel("Gain")
        ax.set_ylabel("el_ampl")
        fig.savefig("test")
        break
                
        print(cost_function_tmp(900, 1, 0, 0.15))




         

    exit()

    # PLOTTING

    # data
    data_dict = read_freq_spectrum_from_pickle(data_path)
    frequencies = data_dict["frequencies"]
    
    # sim
    plt.style.use("retro")
    filename = f"figures/absolute_ampl_calibration/spectra_fit_season{season}_st{station_id}_best_template_fit.pdf"
    pdf = PdfPages(filename)
    
    goodness_of_fit_options = {"reduced_chi2" : calculate_reduced_chi2}
    calculate_gof = goodness_of_fit_options[goodness_of_fit_variable]
    goodness_of_fit_list = []

    template_keys_split = np.array_split(template_keys, 9)
    for channel_id in channel_ids:
        fig, axs = plt.subplots(3, 3, figsize=(20,10))
        axs = np.ndarray.flatten(axs)

        goodness_of_fit_key = {}
        for i, template_key_subset in enumerate(template_keys_split):
            axs[i].plot(frequencies, data_dict["spectrum"][channel_id], label="data", lw=2.)
            trace_tmp = fft.freq2time(data_dict["spectrum"][channel_id], sampling_rate)
            for template_key in template_key_subset:
                fit_result = fit_results_templates[template_key][channel_id]
                fit_param = [param.value for param in fit_result]
#                gain = fit_result[0].value
#                el_ampl = fit_result[1].value
#                el_cst = fit_result[2].value
#                f0 = fit_result[3].value
                fit_function = fit_functions[template_key][channel_id]    
                spectrum_fit = fit_function(frequencies, *fit_param)
                goodness_of_fit = calculate_gof(data_dict["spectrum"][channel_id], spectrum_fit, data_dict["var_spectrum"][channel_id],
                                                      frequencies, freq_range=[0.2, 0.6])
                goodness_of_fit_key[template_key] = goodness_of_fit
                axs[i].plot(frequencies, spectrum_fit, label=f"{template_key}, chi2/dof: {goodness_of_fit:.3f}")
                trace_tmp = fft.freq2time(spectrum_fit, sampling_rate)
            axs[i].legend(fontsize=12, loc="upper right")
            axs[i].set_xlim(0, 1)



        goodness_of_fit_list.append(goodness_of_fit_key)
        fig.suptitle(f"Channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()



    # SAVING BEST FITS PER CHANNEL

    filename_best_fits = filename_base + "_best_fit.csv"

    logging.debug("Saving best fit per channel")
   
    best_fit_results = []
    best_goodness_of_fits = []
    for channel_id in channel_ids:
        goodness_of_fit_key = goodness_of_fit_list[channel_id]
        min_goodness_of_fit_template = min(goodness_of_fit_key, key=goodness_of_fit_key.get)
        best_fit_result = fit_results_templates[min_goodness_of_fit_template][channel_id]
        best_goodness_of_fit = {"best_fit_template" : str(min_goodness_of_fit_template),
                                "gof_method" : goodness_of_fit_variable,
                                "gof_value" : float(min(goodness_of_fit_key.values()))}
        best_fit_results.append(best_fit_result)
        best_goodness_of_fits.append(best_goodness_of_fit)


    value_dicts = [{f.name : f.value for f in fit_result} for fit_result in best_fit_results]
    for i, _ in enumerate(value_dicts):
        value_dicts[i].update(best_goodness_of_fits[i])
    error_dicts = [{f.name : f.error for f in fit_result} for fit_result in best_fit_results]

    value_df = pd.DataFrame(value_dicts)
    error_df = pd.DataFrame(error_dicts)

    header=True
    filename_error = filename_best_fits.split(".csv")[0] + "error" + ".csv"
    value_df.to_csv(os.path.join(save_folder, filename_best_fits), header=header)
    error_df.to_csv(os.path.join(save_folder, filename_error), header=header)


    
    # PLOT BEST TEMPLATE

    plt.style.use("astroparticle_physics")

    representative_channels = [0, 4, 13, 14]
    representative_channel_names = ["PA", "HPol", "LPDA up", "LPDA down"]

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs = np.ndarray.flatten(axs)
    for i, channel_id in enumerate(representative_channels):
        goodness_of_fit_key = goodness_of_fit_list[channel_id]
        min_goodness_of_fit_template = min(goodness_of_fit_key, key=goodness_of_fit_key.get)
        best_fit_result = fit_results_templates[min_goodness_of_fit_template][channel_id]
        fit_param = [param.value for param in best_fit_result]
        print(fit_param)
        fit_function = fit_functions[min_goodness_of_fit_template][channel_id]    
        spectrum_fit = fit_function(frequencies, *fit_param)

        axs[i].plot(frequencies, data_dict["spectrum"][channel_id], label="data", lw=1.)
        axs[i].plot(frequencies, spectrum_fit, label=f"simulation\nGain: {convert_to_db(fit_param[0]):.2f} dB", lw=1.)
        axs[i].set_xlim(0., 1.)
        axs[i].legend(loc="upper right", fontsize=8)
        axs[i].set_title(representative_channel_names[i])


    
    axs[2].set_xlabel("frequencies / GHz")
    axs[3].set_xlabel("frequencies / GHz")
    fig.text(0.04, 0.5, "spectral amplitude / V", va='center', rotation='vertical')
    fig.savefig(f"figures/paper/results_season{season}_st{station_id}_best_fit", dpi=300, bbox_inches="tight")
