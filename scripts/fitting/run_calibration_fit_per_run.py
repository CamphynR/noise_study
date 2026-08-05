"""
Script adapted to quickly get change in gain over the season,
the original script can also be used to get the gain per run but this one is signicantly faster
since it uses the templates found by the default fitting over the full season
it hence does not look for the best template and so only performs one fit!
THIS CODE DOES EXPECT TO FIND A DEFAULT CALIBRATION IN THE FOLDER TO WHICH IT IS SAVED
OR HAVE A DEFAULT FOLDER SPECIFIED IN THE ARGS
"""
import argparse
from astropy.time import Time
import copy
import datetime
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
    This is a helper calibration code, it uses the spectrumFitter class to
    save the callibration coefficients.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help="path to directory containing average spectra per run files,\
                                files are assumed to be of the format average_ft_run*.pickle")
    parser.add_argument("--season", type=str, default=None,
                        help="if None try to infer season from data dir")
    parser.add_argument("--station", "-s", type=int, default=None,
                        help="if None try to infer station from data dir")
    parser.add_argument("--save_dir", default=None,
                        help="base dir to save results, the directory will always contain the subdirectories season????/station??")
    parser.add_argument("--fname_appendix", default=None,
                        help="any files and plots will append this to their name") 
    parser.add_argument("--fit_range", nargs="+", type=float,
                        help="option to set fit range in MHz for testing")
    parser.add_argument("--include_impedance_mismatch", action="store_true")
    parser.add_argument("--perform_plotting", action="store_true", help = "does not save or plot data vs sim spectra, option mainly to save time when fitting a lot of small datasets e.g. to investigate time stability")
    parser.add_argument("--default_calibration_folder", default=None)
    parser.add_argument("--default_calibration_filepath", default=None, help="helper to directly define filepath not ot be used toegther with default_calibration_folder")
    parser.add_argument("--mode", help="choose function to fit mode=constant only fits absolute gain, mode=system_response_weight also fits a slope to the system response")
    parser.add_argument("--ignore_weight", action="store_true", help="legacy to help testing oldr versions")
    args = parser.parse_args()



    # SETTINGS
    # --------------------------------------------------------------------------------------------
    if not "season" in args.data_dir and args.season is None:
        raise KeyError("Cannot infer season from args.data_dir, please specify a season")

    if not "station" in args.data_dir and args.station is None:
        raise KeyError("Cannot infer station_id from args.data_dir, please specify a season")

    if args.season is None:
        season = int(args.data_dir.split("season")[1][:4])
    else:
        if args.season == "2024_partial":
            season = 2024
            season_str = args.season
        elif args.season == "2024_radiant_v2":
            season = 2024
            season_str = args.season
        elif args.season == "2023_lin":
            season = 2023
            season_str = args.season
        else:
            season = int(args.season)
            season_str = args.season

    if args.station is None:
        station_id = int(args.data_dir.split("station")[1][:2])
    else:
        station_id = args.station

    digitizer_version = "digitizer_v3" if season > 2023 \
                  else "digitizer_v2"
    if season_str in ["2024_partial", "2024_radiant_v2"]:
        digitizer_version = "digitizer_v3_resampled"

    sampling_rate = 2.4 * units.GHz if season > 2023 \
            else 3.2 * units.GHz



    if args.save_dir:
        save_folder=f"{args.save_dir}/season{season_str}/station{station_id}/"
        if args.fname_appendix is None:
            save_folder += "default/"
        else:
            save_folder += f"{args.fname_appendix}/"

    else:
        save_folder=f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{season_str}/station{station_id}/"

    
    if args.default_calibration_folder:
        default_folder = os.path.join(args.default_calibration_folder,
                                      f"season{season_str}",
                                      f"station{station_id}",
                                      "default")    
    else:
        default_folder = os.path.join(save_folder,
                                      f"season{season_str}",
                                      f"station{station_id}",
                                      "default")    

    if args.default_calibration_filepath:
        default_filepath = args.default_calibration_filepath
    else:
        default_filepath = os.path.join(default_folder,
                                        f"absolute_amplitude_calibration_season{season_str}_st{station_id}_best_fit.csv")


    default_calibration = pd.read_csv(default_filepath, index_col=0)


    det_time = datetime.datetime(season,8,1)
    if season_str in ["2024_partial", "2024_radiant_v2"]:
        det_time = datetime.datetime(season,6,1)
    det = Detector(signal_chain_measurement_name=None)
    det.update(det_time)



    goodness_of_fit_options = {"reduced_chi2" : calculate_reduced_chi2,
                               "area_diff" : calculate_area_difference}
    goodness_of_fit_variable = "reduced_chi2"

    # MODES
    # the mode determines how the simulation is built
    # "constant" means only the overall gain is fitted
    # LEGACY "electronic_weight" also includes free parameters to fit the electronic noise
    # note constant is preferred

    if args.mode is None:
        # mode = "electronic_weight"
        # mode = "constant"
        # parameter_limits = [(0, None)]

        mode = "system_response_weight"

        # SLOPE IS FIXED TO 2023 SLOPE
        default_filepath_2023_slope = f"absolute_amplitude_results/season2023/station{station_id}/default/absolute_amplitude_calibration_season2023_st{station_id}_best_fit.csv"
        default_calibration_2023_slope = pd.read_csv(default_filepath_2023_slope, index_col=0)

        parameter_guesses = {"gain" : 1000., "slope" : default_calibration_2023_slope["slope"].to_numpy(), "f0" : 0.4}
        parameter_limits = [(0, None), (-3., 3.), (0, 1.)]
        # parameter_fixed = [[False, True, True], [True, False, True], [False, False, True]]
        parameter_fixed = [[False, True, True]]

        # parameter_guesses = None
    elif args.mode == "constant":
        mode = "constant"
        parameter_guesses = None
        parameter_limits = [(0, None)]
        parameter_fixed = None

    if args.fit_range:
        fit_range = [f*units.MHz for f in args.fit_range]
    else:
        fit_range = [0.15, 0.6]
    
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)


    scale_noise_components = None

# ALERT HARDCODED SETTING WATCH OUT !!!!!!!!!!!!!!!!!!!!
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






    # SYSTEM RESPONSE TEMPLATES
    # --------------------------------------------------------------------------------------------
#    system_response_paths = ["sim/library/v2_v3_deep_impulse_responses_for_comparison.json",
#                             "sim/library/v2_v3_surface_impulse_responses.json"]

    # construct a list of all available templates
    system_response_paths = ["sim/library/system_response_templates_deep.json",
                             "sim/library/system_response_templates_surface.json"]



    template_keys = default_calibration["best_fit_template"].to_dict()




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
    data_filenames = sorted([filename for filename in os.listdir(args.data_dir) if "run" in filename])

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



    # START FOR LOOP HERE

    run_nrs = []
    calibration_results = []
    for run_file in data_filenames:
        run_path = os.path.join(args.data_dir, run_file)
        run_nr = int(run_file.split("run")[1].split(".pickle")[0])
        run_nrs.append(run_nr)
        print(f"on run {run_nr}")
        
        save_folder_run = os.path.join(save_folder,
                                       f"default_run{run_nr}")
        if args.fname_appendix:
            save_folder_run += "_" + f"{args.fname_appendix}/"

        os.makedirs(save_folder_run, exist_ok=True)





        print("FITTING")

        if args.ignore_weight:
            weights=None
        else:
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

        system_response = systemResponseTimeDomainIncorporator()
        system_response.begin(response_path=system_response_paths, det=det, station_id=station_id, overwrite_key=template_keys,
                              bandpass_kwargs=bandpass_kwargs,
                              weights=weights)

        fitter = spectrumFitter(run_path,
                                sim_paths,
                                cross_products_path=cross_products_path,
                                sampling_rate=sampling_rate,
                                system_response=system_response,
                                cable_length=cable_length,
                                scale_noise_components=scale_noise_components,
                                impedance_mismatch_factors=impedance_corrections_per_channel)
        fitter.set_fit_range(fit_range)
        fit_results, goodness_of_fit_tmp = fitter.save_fit_results(mode=mode,
                                                                   parameter_guesses=parameter_guesses,
                                                                   parameter_limits=parameter_limits,
                                                                   parameter_fixed=parameter_fixed,
                                                                   save_folder=save_folder_run,
                                                                   filename=filename_base + f"_default_keys.csv")
                                            
        calibration_results.append({"fit_results" : fit_results,
                                    "gof" : goodness_of_fit_tmp})
    
        fit_functions = []
        for channel_id in fitter.channels_to_include:
            channel_function = fitter.get_fit_function(mode=mode, channel_id=channel_id)
            fit_functions.append(channel_function)

        
        # the fitter reads the sim config
        channel_ids = fitter.channels_to_include



        settings_dict = {
                "season" : season,
                "station" : station_id,
                "mode" : mode,
                "parameter_guesses" : parameter_guesses,
                "parameter_limits" : parameter_limits,
                "parameter_fixed" : parameter_fixed,
                "digitizer_version" : digitizer_version,
                "goodness_of_fit_variable" : goodness_of_fit_variable,
                "fit_range" : fit_range,
                "sim_paths" : sim_paths,
                "response_templates_used" : template_keys,
                "channels_to_include" : [int(c) for c in channel_ids],
                "scale_noise_components" : scale_noise_components
                }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        settings_path = os.path.join(save_folder_run, "fit_settings.json")
        with open(settings_path, "w") as settings_file:
            json.dump(settings_dict, settings_file, indent=4, cls=NumpyEncoder)




        if args.perform_plotting:
            # PLOTTING
            # --------------------------------------------------------------------------------------------
            # PLOTTING ALL TEMPLATES
            # ----------------------


            print("PLOTTING DEFAULT TEMPLATES")
            # data
            data_dict = read_freq_spectrum_from_pickle(data_path)
            frequencies = data_dict["frequencies"]
            

            plot_data_tmpl_fname = filename_base
            plot_data_tmpl_fname += "_plot_data.pickle"
            plot_data_tmpl_path = os.path.join(save_folder, plot_data_tmpl_fname)
            plot_data_tmpl = {"channel_ids" : channel_ids,
                              "frequencies" : frequencies,
                              "data" : [[] for ch_id in channel_ids],
                              "sim" : [[] for ch_id in channel_ids]}


            # sim
            plt.style.use("retro")
            filename = f"figures/absolute_ampl_calibration/spectra_fit_season{season_str}_st{station_id}_default_template_fit"
            if args.fname_appendix is not None:
                filename += "_" + args.fname_appendix
            filename += ".pdf"
            pdf = PdfPages(filename)
            
            calculate_gof = goodness_of_fit_options[goodness_of_fit_variable]
            goodness_of_fit_list = []

            for ch_i, channel_id in enumerate(channel_ids):
                fig, ax = plt.subplots()

                data_spectrum = data_dict["spectrum"][channel_id]
                data_var = data_dict["var_spectrum"][channel_id]
                data_line = ax.plot(frequencies, data_dict["spectrum"][channel_id], label="data", lw=2.)
                ax.fill_between(frequencies[(0.15 < frequencies)&(frequencies<0.6)],
                                data_spectrum[(0.15 < frequencies)&(frequencies<0.6)] - data_var[(0.15 < frequencies)&(frequencies<0.6)],
                                data_spectrum[(0.15 < frequencies)&(frequencies<0.6)] + data_var[(0.15 < frequencies)&(frequencies<0.6)],
                                color=data_line[0].get_color(),
                                alpha=0.3)
                plot_data_tmpl["data"][ch_i] = data_dict["spectrum"][channel_id]



                fit_result = fit_results[ch_i]
                fit_param = [param.value for param in fit_result]
        #                gain = fit_result[0].value
        #                el_ampl = fit_result[1].value
        #                el_cst = fit_result[2].value
        #                f0 = fit_result[3].value
                fit_function = fit_functions[ch_i]    
                spectrum_fit = fit_function(frequencies, *fit_param)
                goodness_of_fit = calculate_gof(data_dict["spectrum"][channel_id], spectrum_fit, data_dict["var_spectrum"][channel_id],
                                                      frequencies, freq_range=fitter.fit_range)
                ax.plot(frequencies, spectrum_fit, label=f"{template_keys[channel_id]},\nG={convert_to_db(fit_result[0].value):.2f} dB\nchi2/dof: {goodness_of_fit:.3f}")
                plot_data_tmpl["sim"][ch_i] = spectrum_fit
                ax.legend(fontsize=12, loc="upper right")
                ax.set_xlim(0, 1)



                goodness_of_fit_list.append(goodness_of_fit)
                fig.suptitle(f"Channel {channel_id}")
                fig.tight_layout()
                fig.savefig(pdf, format="pdf")
                plt.close(fig)

            pdf.close()
            del pdf


            with open(plot_data_tmpl_path, "wb") as plot_data_tmpl_file:
                pickle.dump(plot_data_tmpl,
                            plot_data_tmpl_file)






            

    all_run_name = f"season{season_str}_st{station_id}_all_runs_compiled"
    if args.fname_appendix:
        all_run_name += "_" + args.fname_appendix
    all_run_name += ".pickle"

    all_run_save_path = os.path.join(save_folder,
                                     all_run_name)
    with open(all_run_save_path, "wb") as file:
        save_dict = {"run_nr" : run_nrs,
                "calibration" :calibration_results}
        pickle.dump(save_dict, file)
