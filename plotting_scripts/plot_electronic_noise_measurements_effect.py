import argparse
from astropy.time import Time
import json
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import pandas as pd

from NuRadioReco.detector.RNO_G.rnog_detector import Detector


from fitting.spectrumFitter import spectrumFitter
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator





if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--station", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    
    det = Detector()
    det_time = Time(f"{args.season}-08-01")
    det.update(det_time)

    sampling_rate = det.get_sampling_frequency(station_id=args.station)
    channel_ids = sorted(det.get_channel_ids(station_id=args.station))


    if args.season < 2024:
        digitizer_version = 2
    else:
        if args.station == 24:
            digitizer_version = 2
        else:
            digitizer_version = 3


    fitting_mode = "electronic_temp_cross"
    include_impedance_mismatch_correction = True
    fit_range = [0.1, 0.6]
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)


    channel_mapping = {
        "deep" : [0, 1, 2, 3],
        "helper" : [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
        "surface" : [12, 13, 14, 15, 16, 17, 18, 19, 20]
        }


    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{args.season}/station{args.station}/clean/average_ft_combined.pickle"

    sim_dir_electronic_measured = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise/digitizer_v{digitizer_version}"
    sim_paths_electronic_measured = [os.path.join(sim_dir_electronic_measured, f"{component}/station{args.station}/clean/average_ft.pickle")
                 for component in ["ice", "electronic", "galactic"]]
    cross_products_path_electronic_measured = f"{sim_dir_electronic_measured}/cross_products/station{args.station}/cross_products.pickle"


    sim_dir_electronic_fit = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response/digitizer_v{digitizer_version}"
    sim_paths_electronic_fit = [os.path.join(sim_dir_electronic_fit, f"{component}/station{args.station}/clean/average_ft.pickle")
                 for component in ["ice", "electronic", "galactic"]]
    cross_products_path_electronic_fit = f"{sim_dir_electronic_fit}/cross_products/station{args.station}/cross_products.pickle"

    
    save_folder_eletronic_fit = "/user/rcamphyn/noise_study/absolute_amplitude_results/no_measured_electronic_noise"

    
    # we perform the fitting again for fitted electronic noise

    system_response_paths = ["sim/library/deep_templates_combined.json",
                             "sim/library/v2_v3_surface_impulse_responses.json"]
    with open(system_response_paths[0], "r") as f:
        deep_keys = list(json.load(f).keys())
    deep_keys.remove("time")
    # v3_ch5 was a test and does not contain a physical template
    template_keys = deep_keys

    with open(system_response_paths[1], "r") as f:
        surface_keys = list(json.load(f).keys())
    surface_keys += ["surface_query"]
    surface_keys.remove("time")
    surface_keys.remove("v3_ch5")
    template_keys += surface_keys

    
    fit_results_templates = {}
    fit_functions = {}



    filename_base = f"absolute_amplitude_calibration_electronic_fit_season{args.season}_st{args.station}"
    goodness_of_fits_dict = {}

    for template_key in template_keys:
        system_response = systemResponseTimeDomainIncorporator()
        system_response.begin(det=0, response_path=system_response_paths, overwrite_key=template_key,
                              bandpass_kwargs=bandpass_kwargs)

        fitter = spectrumFitter(data_path,
                                sim_paths_electronic_fit,
                                cross_products_path=cross_products_path_electronic_fit,
                                sampling_rate=sampling_rate,
                                system_response=system_response,
                                include_impedance_mismatch_correction=include_impedance_mismatch_correction)
        fitter.set_fit_range(fit_range)
        fit_functions_template = []
        for channel_id in channel_ids:
            channel_function = fitter.get_fit_function(mode=fitting_mode, channel_id=channel_id)
            fit_functions_template.append(channel_function)
        fit_functions[template_key] = fit_functions_template
        if not os.listdir(save_folder_eletronic_fit) or args.overwrite:
            fit_results, goodness_of_fits = fitter.save_fit_results(mode=fitting_mode, save_folder=save_folder_eletronic_fit,
                                                  filename=filename_base + f"_key{template_key}.csv")
            fit_results_templates[template_key] = fit_results
            goodness_of_fits_dict[template_key] = goodness_of_fits




    if not os.listdir(save_folder_eletronic_fit) or args.overwrite:
        # SAVING BEST FITS PER CHANNEL

        filename_best_fits = filename_base + "_best_fit.csv"

        logging.debug("Saving best fit per channel")

        best_fit_results = []
        best_goodness_of_fits = []
        for channel_id in channel_ids:
            if channel_id in channel_mapping["deep"] or channel_id in channel_mapping["helper"]:
                subset = deep_keys
            else:
                subset = surface_keys
            goodness_of_fit_subset = {key : goodness_of_fits_dict[key][channel_id] for key in subset}
            min_goodness_of_fit_template = min(goodness_of_fit_subset, key=goodness_of_fit_subset.get)
            best_fit_result = fit_results_templates[min_goodness_of_fit_template][channel_id]
            best_goodness_of_fit = {"best_fit_template" : str(min_goodness_of_fit_template),
                                    "gof_method" : fitter.goodness_of_fit_function.__name__,
                                    "gof_value" : float(min(goodness_of_fit_subset.values()))}
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
            value_df.to_csv(os.path.join(save_folder_eletronic_fit, filename_best_fits), header=header)
            error_df.to_csv(os.path.join(save_folder_eletronic_fit, filename_error), header=header)



    # load in fit parameters

    save_folder_electronic_measured = "/user/rcamphyn/noise_study/absolute_amplitude_results"
    fit_parameters_electronic_measured = pd.read_csv(f"{save_folder_electronic_measured}/absolute_amplitude_calibration_season{args.season}_st{args.station}_best_fit.csv").values
    
    fit_parameters_electronic_fit = pd.read_csv(f"{save_folder_eletronic_fit}/absolute_amplitude_calibration_electronic_fit_season{args.season}_st{args.station}_best_fit.csv").values 



    # PLOTTING
    plt.style.use("astroparticle_physics")
    plotname = "figures/electronic_noise_fit_vs_measured.pdf"
    pdf = PdfPages(plotname)
    for channel_id in channel_ids:
        fig, ax = plt.subplots()


        
        system_response = systemResponseTimeDomainIncorporator()
        system_response.begin(det=0, response_path=system_response_paths, overwrite_key=fit_parameters_electronic_measured[channel_id][5],
                              bandpass_kwargs=bandpass_kwargs)
        fitter_measured = spectrumFitter(data_path,
                                sim_paths_electronic_measured,
                                cross_products_path=cross_products_path_electronic_measured,
                                sampling_rate=sampling_rate,
                                system_response=system_response,
                                include_impedance_mismatch_correction=include_impedance_mismatch_correction)
        fitter_measured.set_fit_range(fit_range)
        fit_functions_measured = fitter_measured.get_fit_function(mode=fitting_mode, channel_id=channel_id)



        fit_function = fit_functions[fit_parameters_electronic_fit[channel_id][5]][channel_id]
        frequencies = fitter.frequencies
        data = fitter.data_spectrum
        
        sim_spectrum_electronic_measured = fit_functions_measured(frequencies, *fit_parameters_electronic_measured[channel_id][1:5])
        sim_spectrum_electronic_fit = fit_function(frequencies, *fit_parameters_electronic_fit[channel_id][1:5])

        ax.plot(frequencies, data[channel_id], label="data")
        ax.plot(frequencies, sim_spectrum_electronic_measured, label=f"measured electronic noise\nGain={fit_parameters_electronic_measured[channel_id][1]:.2f}\nTemplate: {fit_parameters_electronic_measured[channel_id][5]}")
        ax.plot(frequencies, sim_spectrum_electronic_fit, label=f"fitted electronic noise\nGain={fit_parameters_electronic_fit[channel_id][1]:.2f}\nTemplate: {fit_parameters_electronic_fit[channel_id][5]}")
        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("spectral amplitude / V")
        ax.set_title(f"Channel {channel_id}")
        ax.set_xlim(0, 1.)
        ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
