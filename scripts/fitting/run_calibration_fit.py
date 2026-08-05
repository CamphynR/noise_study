import argparse
from astropy.time import Time
import copy
import json
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import stats
from scipy.interpolate import interp1d

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import fft, units

from fitting.spectrumFitter import spectrumFitter
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle, convert_to_db


# GOF FUNCTIONS USED TO SELECT A RESPONSE TEMPLATE
# note that the cost function of the fit itself is always Minuit's LeastSquares,
# the cost function can be changed in the spectrumFitter's begin function 
def calculate_reduced_chi2(spectrum_1, spectrum_2, var_1, frequencies, freq_range):
    mask = (frequencies > freq_range[0]) & (frequencies < freq_range[1])
    spectrum_1 = np.abs(spectrum_1[mask])
    spectrum_2 = np.abs(spectrum_2[mask])
    var_1 = var_1[mask]
    ndof = len(spectrum_1)

    chi2 = np.sum(((spectrum_1 - spectrum_2)**2/var_1))
    chi2_reduced = chi2/ndof

    return chi2_reduced


def calculate_area_difference(spectrum_1, spectrum_2, var_1, frequencies, freq_range):
    mask = (frequencies > freq_range[0]) & (frequencies < freq_range[1])
    spectrum_1 = np.abs(spectrum_1[mask])
    spectrum_2 = np.abs(spectrum_2[mask])
    area_1 = np.trapezoid(spectrum_1)
    area_2 = np.trapezoid(spectrum_2)
    return np.abs(area_1 - area_2)



if __name__ == "__main__":
    """
    This is the master calibration code, it uses the spectrumFitter class to
    save the callibration coefficients.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None,
                        help="direct path to file containing data spectra,\
                                usefull for testing and calculating gain per run,\
                                if not specified the code will default to the\
                                average spectrum over the given season")
    parser.add_argument("--sim_dir", default=None)
    parser.add_argument("--season", type=str, default=2024)
    parser.add_argument("--station", "-s", type=int, default=12)
    parser.add_argument("--save_dir", default=None,
                        help="base dir to save results, the directory will always contain the subdirectories season????/station??")
    parser.add_argument("--fname_appendix", default=None,
                        help="any files and plots will append this to their name") 
    parser.add_argument("--fit_range", nargs="+", type=float,
                        help="option to set fit range in MHz for testing")
    parser.add_argument("--index",
                        help="optional test option when testing different antenna models")
    parser.add_argument("--include_impedance_mismatch", action="store_true")
    parser.add_argument("--skip_data_saving", action="store_true", help = "does not save plot data, option mainly to save time when fitting a lot of small datasets e.g. to investigate time stability")
    args = parser.parse_args()


    # SETTINGS
    # --------------------------------------------------------------------------------------------
    # these are saved in a json file for posterity
    channel_mapping = {
        "deep" : [0, 1, 2, 3],
        "helper" : [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
        "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
        }


    if args.season == "2024_partial":
        season = 2024
        season_str = "2024_partial"
    elif args.season == "2024_radiant_v2":
        season = 2024
        season_str = args.season
    elif args.season == "2023_lin":
        season = 2023
        season_str = args.season
    else:
        season = int(args.season)
        season_str = args.season


    if args.save_dir:
        save_folder=f"{args.save_dir}/season{season_str}/station{args.station}/"
    else:
        save_folder=f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season_str}/station{args.station}/"
    if args.fname_appendix is None:
        save_folder += "default/"
    else:
        save_folder += f"{args.fname_appendix}/"

    os.makedirs(save_folder, exist_ok=True)

    det_time = Time(f"{season}-08-01")
    if season_str == "2024_partial" or season_str == "2024_radiant_v2":
        det_time = Time(f"{season}-06-01")
    det = Detector(signal_chain_measurement_name=None)
    det.update(det_time)
    station_info = det.get_station(station_id=args.station)

    digitizer_version = "digitizer_v3" if season > 2023 \
                  else "digitizer_v2"
    if season_str == "2024_partial" or season_str == "2024_radiant_v2":
        digitizer_version = "digitizer_v3_resampled"
    station_id = args.station
    sampling_rate = 2.4 * units.GHz if season > 2023 \
            else 3.2 * units.GHz


    goodness_of_fit_options = {"reduced_chi2" : calculate_reduced_chi2,
                               "area_diff" : calculate_area_difference}
    goodness_of_fit_variable = "reduced_chi2"

    # MODES
    # the mode determines how the simulation is built
    # "constant" means only the overall gain is fitted
    # LEGACY "electronic_weight" also includes free parameters to fit the electronic noise
    # note constant is preferred


    # TESTING SLOPE EFFECT


    # default_filepath = f"absolute_amplitude_results/season2023/station{args.station}/default/absolute_amplitude_calibration_season2023_st{args.station}_best_fit.csv"
    # default_calibration = pd.read_csv(default_filepath, index_col=0)

    # print("--------------------------------------------------")
    # print("YOU ARE USING SLOPE 2023)")
    # print("--------------------------------------------------")
    # mode = "system_response_weight"
    # parameter_limits = [(0, None), (-3., 3.), (0., 1.)]
    # parameter_fixed = [[False, True, True]]
    # parameter_guesses = {"gain" : 1000., "slope" : default_calibration["slope"].to_numpy(), "f0" : 0.4}

    print("ATTENTION YOU ARE FITTING SLOPE TO SYSTEM RESPONSE")
    mode = "system_response_weight"
    parameter_limits = [(0, None), (-3., 3.), (0., 1.)]
    parameter_fixed = [[False, True, True], [True, False, True], [False, False, True]]
    parameter_guesses = {"gain" : 1000., "slope" : 0., "f0" : 0.4}
    

    # mode = "constant"
    # parameter_limits = [(0, None)]

    if args.fit_range:
        fit_range = [f*units.MHz for f in args.fit_range]
    else:
        fit_range = [0.15, 0.6]
    
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)


    scale_noise_components = None

# ALERT HARDCODED SETTING WATCH OUT !!!!!!!!!!!!!!!!!!!!
# ======================================================

# TESTING GALACTIC EFFECT
#    galactic_scale_per_antenna_type_gsm2016 = {"VPols" : 0.96, "HPols" : 1.01, "LPDAs" : 0.975}
#    galactic_scale_per_antenna_type_lfss = {"VPols" : 1.07, "HPols" : 1.07, "LPDAs" : 1.07}
#    antenna_types = {
#            "VPols" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
#            "HPols" : [4, 8, 11, 21],
#            "LPDAs" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
#            }
#
#    galactic_scaling = np.ones(24)
#    for antenna_type, channel_ids in antenna_types.items():
#        for channel_id in channel_ids:
#            galactic_scaling[channel_id] *= galactic_scale_per_antenna_type_lfss[antenna_type]
#
#    scale_noise_components = {
#            "ice" : np.ones(24),
#            "electronic" : np.ones(24),
#            "galactic" : galactic_scaling
#            }





    if args.include_impedance_mismatch:
        correction_factor_dir = "sim/library/impedance_correction_factors"
        antenna_types = {
            "VPol_100m" : [0, 1, 2, 3, 9, 10, 22, 23],
            "VPol_80m" : [5],
            "VPol_60m" : [6],
            "VPol_40m" : [7],
                }

        impedance_correction_per_antenna_paths = {
            "VPol_100m" : "RNOG_vpol_v4_final_IGLU_n1.70_correction_factor.npz",
            "VPol_80m" : "RNOG_vpol_v4_final_IGLU_n1.70_correction_factor.npz",
            "VPol_60m" : "RNOG_vpol_v4_final_IGLU_n1.70_correction_factor.npz",
            "VPol_40m" : "RNOG_vpol_v4_final_IGLU_n1.60_correction_factor.npz",
                }

        impedance_corrections = {}
        for antenna_type, path in impedance_correction_per_antenna_paths.items():
            with np.load(os.path.join(correction_factor_dir, path)) as content:
                frequencies_imp = content["frequencies"]
                f_correction = content["f_correction"]
                impedance_corrections[antenna_type] = interp1d(frequencies_imp,
                                                               np.abs(f_correction),
                                                               bounds_error=False,
                                                               fill_value=1.)

        impedance_corrections_per_channel = [None for _ in range(24)]
        for antenna_type, channel_ids in antenna_types.items():
            for channel_id in channel_ids:
                impedance_corrections_per_channel[channel_id] = impedance_corrections[antenna_type]

    else:
        impedance_corrections_per_channel = None
        


    # DATA
    # --------------------------------------------------------------------------------------------
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{season_str}/station{station_id}/clean/average_ft_combined.pickle"
    if args.sim_dir:
        sim_dir = args.sim_dir
    else:
        sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/default/{digitizer_version}"
#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise_new_impedance_mismatch_antenna_v4/{digitizer_version}"
#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/antenna_model_variations/vpol_v4_n1.{args.index}0"
#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/galaxy_model_variation/lfss"
#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/antenna_model_variations/vpol_v4_shift_{args.index}MHz"
#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/test_bandwidth"
    sim_paths = [os.path.join(sim_dir, f"{component}/station{station_id}/clean/average_ft.pickle")
                 for component in ["ice", "electronic", "galactic"]]

#    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response/{digitizer_version}"
#    sim_paths = [os.path.join(sim_dir, f"{component}/station{station_id}/clean/average_ft.pickle")
#                 for component in ["ice", "electronic", "galactic"]]


    cross_products_path = f"{sim_dir}/cross_products/station{station_id}/cross_products.pickle"

    cable_length=11




    # SYSTEM RESPONSE TEMPLATES
    # --------------------------------------------------------------------------------------------
#    system_response_paths = ["sim/library/v2_v3_deep_impulse_responses_for_comparison.json",
#                             "sim/library/v2_v3_surface_impulse_responses.json"]

    # construct a list of all available templates
    system_response_paths = ["sim/library/system_response_templates_deep.json",
                             "sim/library/system_response_templates_surface.json"]
    template_keys = []

    with open(system_response_paths[0], "r") as f:
        deep_keys = list(json.load(f).keys())
    deep_keys.remove("time")
    template_keys = copy.deepcopy(deep_keys)

    with open(system_response_paths[1], "r") as f:
        surface_keys = list(json.load(f).keys())
    surface_keys += ["surface_query"]
    surface_keys.remove("time")
    # v3_ch5 was a test and does not contain a physical template
    surface_keys.remove("v3_ch5")
    template_keys += copy.deepcopy(surface_keys)


    # for the fitting we do not mix RADIANT v2 and v3 templates
    # this can be commented for testing purposes to see how sensitive
    # the fit is to RADIANT versions


    print("DO NOT FORGET YOU ARE FIXING RADIANT VERSIONS")
    template_keys_copy = copy.deepcopy(template_keys)
    if season > 2023 and not season_str == "2024_partial" and not season_str == "2024_radiant_v2":
        for key in template_keys_copy:
            if not key.startswith("v3"):
                try:
                    deep_keys.remove(key)
                except:
                    surface_keys.remove(key)
                template_keys.remove(key)
    elif season <= 2023  or season_str == "2024_partial" or season_str == "2024_radiant_v2":
        for key in template_keys_copy:
            if key.startswith("v3"):
                try:
                    deep_keys.remove(key)
                except:
                    surface_keys.remove(key)
                template_keys.remove(key)



    print("FITTING")


    # TESTING WEIGHTS BY IMPULSE RESPONSE
    from modules.pulserResponse import pulserResponse
    pulser_helper = pulserResponse()
    frequencies_pulser = fft.freqs(2048, sampling_rate=sampling_rate)
    pulser_response = pulser_helper(frequencies_pulser)
    weights = {ch : 1./pulser_response for ch in np.arange(24)}

    print("DO NOT FORGET YOU ARE DIVIDING PULSER")

    

    # FITTING
    # --------------------------------------------------------------------------------------------
    fit_results_templates = {}
    fit_functions = {}


    filename_base = f"absolute_amplitude_calibration_season{season_str}_st{station_id}"
    if args.fname_appendix is not None:
        filename_base += "_" + args.fname_appendix

    for template_key in template_keys:
        system_response = systemResponseTimeDomainIncorporator()
        #station_id here is just to get settings like sampling_rate and nr_samples
        system_response.begin(response_path=system_response_paths, det=det, station_id=11 ,overwrite_key=template_key,
                              bandpass_kwargs=bandpass_kwargs,
                              weights = weights)

        fitter = spectrumFitter(data_path,
                                sim_paths,
                                cross_products_path=cross_products_path,
                                sampling_rate=sampling_rate,
                                system_response=system_response,
                                cable_length=cable_length,
                                scale_noise_components=scale_noise_components,
                                impedance_mismatch_factors=impedance_corrections_per_channel)
        fitter.set_fit_range(fit_range)
        fit_results, goodness_of_fit_tmp = fitter.save_fit_results(mode=mode,
                                                                   parameter_limits=parameter_limits,
                                                                   parameter_guesses=parameter_guesses,
                                                                   parameter_fixed=parameter_fixed,
                                                                   save_folder=save_folder,
                                                                   filename=filename_base + f"_key{template_key}.csv")
        fit_results_templates[template_key] = fit_results

        fit_functions_template = []
        for channel_id in fitter.channels_to_include:
            channel_function = fitter.get_fit_function(mode=mode, channel_id=channel_id)
            fit_functions_template.append(channel_function)
        fit_functions[template_key] = fit_functions_template

    
    # the fitter reads the sim config
    channel_ids = fitter.channels_to_include


         



    # PLOTTING
    # --------------------------------------------------------------------------------------------
    # PLOTTING ALL TEMPLATES
    # ----------------------


    print("PLOTTING ALL TEMPLATES")
    # data
    data_dict = read_freq_spectrum_from_pickle(data_path)
    frequencies = data_dict["frequencies"]
    

    plot_data_tmpl_fname = filename_base
    plot_data_tmpl_fname += "_plot_data_all_templates.pickle"
    plot_data_tmpl_path = os.path.join(save_folder, plot_data_tmpl_fname)
    plot_data_tmpl = {"channel_ids" : channel_ids,
                      "frequencies" : frequencies,
                      "data" : {key : [[] for ch_id in channel_ids] for key in template_keys},
                      "sim" : {key : [[] for ch_id in channel_ids] for key in template_keys}}


    # sim
    plt.style.use("retro")
    filename = f"figures/absolute_ampl_calibration/spectra_fit_season{season_str}_st{station_id}_all_template_fit"
    if args.fname_appendix is not None:
        filename += "_" + args.fname_appendix
    filename += ".pdf"
    pdf = PdfPages(filename)
    
    calculate_gof = goodness_of_fit_options[goodness_of_fit_variable]
    goodness_of_fit_list = []

    template_keys_split = np.array_split(template_keys, 9)
    for ch_i, channel_id in enumerate(channel_ids):
        fig, axs = plt.subplots(3, 3, figsize=(20,10), sharex=True, sharey=True)
        axs = np.ndarray.flatten(axs)

        goodness_of_fit_key = {}
        for i, template_key_subset in enumerate(template_keys_split):
            axs[i].plot(frequencies, data_dict["spectrum"][channel_id], label="data", lw=2.)
            trace_tmp = fft.freq2time(data_dict["spectrum"][channel_id], sampling_rate)
            for template_key in template_key_subset:
                plot_data_tmpl["data"][template_key][ch_i] = data_dict["spectrum"][channel_id]



                fit_result = fit_results_templates[template_key][ch_i]
                fit_param = [param.value for param in fit_result]
#                gain = fit_result[0].value
#                el_ampl = fit_result[1].value
#                el_cst = fit_result[2].value
#                f0 = fit_result[3].value
                fit_function = fit_functions[template_key][ch_i]    
                spectrum_fit = fit_function(frequencies, *fit_param)
                goodness_of_fit = calculate_gof(data_dict["spectrum"][channel_id], spectrum_fit, data_dict["var_spectrum"][channel_id],
                                                      frequencies, freq_range=fitter.fit_range)
                goodness_of_fit_key[template_key] = goodness_of_fit
                axs[i].plot(frequencies, spectrum_fit, label=f"{template_key}, chi2/dof: {goodness_of_fit:.3f}")
                plot_data_tmpl["sim"][template_key][ch_i] = spectrum_fit
                trace_tmp = fft.freq2time(spectrum_fit, sampling_rate)
            axs[i].legend(fontsize=12, loc="upper right")
            axs[i].set_xlim(0, 1)



        goodness_of_fit_list.append(goodness_of_fit_key)
        fig.suptitle(f"Channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
    del pdf


    if not args.skip_data_saving:
        with open(plot_data_tmpl_path, "wb") as plot_data_tmpl_file:
            pickle.dump(plot_data_tmpl,
                        plot_data_tmpl_file)

    
    print("PLOTTING BEST TEMPLATE")

    # PLOTTING BEST TEMPLATE
    # ----------------------

    # we also save the plot data for comparison to other fit / simulation methods
    plot_data_fname = filename_base
    plot_data_fname += "_plot_data.pickle"
    plot_data_path = os.path.join(save_folder, plot_data_fname)
    plot_data = {"channel_ids" : channel_ids,
                 "frequencies" : frequencies,
                 "data" : [[] for channel_id in channel_ids],
                 "sim" : [[] for channel_id in channel_ids]}


    pdf_name = f"figures/absolute_ampl_calibration/spectra_fit_season{season_str}_st{station_id}_best_template_fit"
    if args.fname_appendix is not None:
        pdf_name += "_" + args.fname_appendix
    pdf_name += ".pdf"
    pdf = PdfPages(pdf_name)

    for i, channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots()
        goodness_of_fit_key = goodness_of_fit_list[i]
        if channel_id in channel_mapping["deep"] or channel_id in channel_mapping["helper"]:
            subset = deep_keys
        else:
            subset = surface_keys
        goodness_of_fit_key_subset = {key : goodness_of_fit_key[key] for key in subset}
        min_goodness_of_fit_template = min(goodness_of_fit_key_subset, key=goodness_of_fit_key_subset.get)
        best_fit_result = fit_results_templates[min_goodness_of_fit_template][i]
        fit_param = [param.value for param in best_fit_result]
        fit_function = fit_functions[min_goodness_of_fit_template][i]    
        spectrum_fit = fit_function(frequencies, *fit_param)

        data_spectrum = data_dict["spectrum"][channel_id]
        data_var = data_dict["var_spectrum"][channel_id]
        data_line = ax.plot(frequencies, data_spectrum, label="data", lw=1.)
        ax.fill_between(frequencies[(0.15 < frequencies)&(frequencies<0.6)],
                        data_spectrum[(0.15 < frequencies)&(frequencies<0.6)] - data_var[(0.15 < frequencies)&(frequencies<0.6)],
                        data_spectrum[(0.15 < frequencies)&(frequencies<0.6)] + data_var[(0.15 < frequencies)&(frequencies<0.6)],
                        color=data_line[0].get_color(),
                        alpha=0.3)
        plot_data["data"][i] = data_spectrum


        ax.plot(frequencies, spectrum_fit, label=f"simulation\nGain:\n{convert_to_db(fit_param[0]):.2f} dB\nTemplate:\n{min_goodness_of_fit_template}", lw=1.)
        plot_data["sim"][i] = spectrum_fit

        ax.axvspan(*fit_range, alpha=0.2, label = "fit range", edgecolor="black", linestyle="dashed")

        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("spectral amplitude / V")
        ax.set_xlim(0., 1.)
        ax.legend(loc="upper right", fontsize=16)
        ax.set_title(f"Channel {channel_id}")

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
    del pdf


    with open(plot_data_path, "wb") as plot_data_file:
        pickle.dump(plot_data,
                    plot_data_file)



    print("SAVING")

    # SAVING BEST FITS PER CHANNEL

    filename_best_fits = filename_base
    filename_best_fits += "_best_fit.csv"

    logging.debug("Saving best fit per channel")

    best_fit_results = []
    best_goodness_of_fits = []
    for i, channel_id in enumerate(channel_ids):
        goodness_of_fit_key = goodness_of_fit_list[i]
        if channel_id in channel_mapping["deep"] or channel_id in channel_mapping["helper"]:
            subset = deep_keys
        else:
            subset = surface_keys
        goodness_of_fit_key_subset = {key : goodness_of_fit_key[key] for key in subset}
        min_goodness_of_fit_template = min(goodness_of_fit_key_subset, key=goodness_of_fit_key_subset.get)
        best_fit_result = fit_results_templates[min_goodness_of_fit_template][i]
        best_goodness_of_fit = {"best_fit_template" : str(min_goodness_of_fit_template),
                                "gof_method" : goodness_of_fit_variable,
                                "gof_value" : float(min(goodness_of_fit_key_subset.values()))}
        best_fit_results.append(best_fit_result)
        best_goodness_of_fits.append(best_goodness_of_fit)


    value_dicts = [{f.name : f.value for f in fit_result} for fit_result in best_fit_results]
    for i, _ in enumerate(value_dicts):
        value_dicts[i].update(best_goodness_of_fits[i])
    error_dicts = [{f.name : f.error for f in fit_result} for fit_result in best_fit_results]

    value_df = pd.DataFrame(value_dicts)
    value_df["channel_id"] = channel_ids
    value_df.set_index("channel_id", inplace=True)

    error_df = pd.DataFrame(error_dicts)
    error_df["channel_id"] = channel_ids
    error_df.set_index("channel_id", inplace=True)

    header=True
    filename_error = filename_best_fits.split(".csv")[0] + "error" + ".csv"
    value_df.to_csv(os.path.join(save_folder, filename_best_fits), header=header)
    error_df.to_csv(os.path.join(save_folder, filename_error), header=header)
    


    # PLOT REPRESENTATIVE CHANNELS FOR PAPER

    plt.style.use("astroparticle_physics")

    representative_channels = [0, 4, 13, 14]
    representative_channel_names = ["PA", "HPol", "LPDA up", "LPDA down"]

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(15, 10))
    axs = np.ndarray.flatten(axs)
    for i, channel_id in enumerate(representative_channels):
        goodness_of_fit_key = goodness_of_fit_list[i]
        if channel_id in channel_mapping["deep"] or channel_id in channel_mapping["helper"]:
            subset = deep_keys
        else:
            subset = surface_keys
        goodness_of_fit_key_subset = {key : goodness_of_fit_key[key] for key in subset}
        min_goodness_of_fit_template = min(goodness_of_fit_key_subset, key=goodness_of_fit_key_subset.get)
        best_fit_result = fit_results_templates[min_goodness_of_fit_template][i]
        fit_param = [param.value for param in best_fit_result]
        print(fit_param)
        fit_function = fit_functions[min_goodness_of_fit_template][i]    
        spectrum_fit = fit_function(frequencies, *fit_param)

        axs[i].plot(frequencies, data_dict["spectrum"][channel_id], label="data", lw=1.)
#        axs[i].plot(frequencies, spectrum_fit, label=f"simulation\nGain:\n{convert_to_db(fit_param[0]):.2f} dB\nTemplate:\n{min_goodness_of_fit_template}", lw=1.)
        axs[i].plot(frequencies, spectrum_fit, label=f"simulation\nGain:\n{convert_to_db(fit_param[0]):.2f} dB", lw=1.)
        axs[i].axvspan(*fit_range, alpha=0.2, label = "fit range", edgecolor="black", linestyle="dashed")
        axs[i].set_xlim(0., 1.)
        axs[i].legend(loc="upper right", fontsize=16)
        axs[i].set_title(representative_channel_names[i])


    
    axs[2].set_xlabel("frequencies / GHz")
    axs[3].set_xlabel("frequencies / GHz")
    fig.text(-0.04, 0.5, "spectral amplitude / V", va='center', rotation='vertical')
    fig.tight_layout()
    figname = f"figures/paper/results_season{season_str}_st{station_id}_best_fit"
    if args.fname_appendix is not None:
        figname += "_" + args.fname_appendix
    figname+=".png"
    fig.savefig(figname, dpi=300, bbox_inches="tight")






    settings_dict = {
            "season" : season,
            "station" : station_id,
            "mode" : mode,
            "parameter_limits" : parameter_limits,
            "parameter_guesses" : parameter_guesses,
            "parameter_fixed" : parameter_fixed,
            "digitizer_version" : digitizer_version,
            "goodness_of_fit_variable" : goodness_of_fit_variable,
            "fit_range" : fit_range,
            "sim_paths" : sim_paths,
            "response_templates_used" : template_keys,
            "channels_to_include" : [int(c) for c in channel_ids],
            "scale_noise_components" : scale_noise_components,
            }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    settings_path = os.path.join(save_folder, "fit_settings.json")
    with open(settings_path, "w") as settings_file:
        json.dump(settings_dict, settings_file, indent=4, cls=NumpyEncoder)
