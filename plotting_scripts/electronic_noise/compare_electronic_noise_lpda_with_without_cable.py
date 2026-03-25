import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd



from fitting.spectrumFitter import spectrumFitter
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_freq_spectrum_from_pickle







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--station", type=int, default=11)
    args = parser.parse_args()
    
    channel_ids = np.arange(24)
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)

    if args.season > 2023:
        digitizer_version = "digitizer_v3"
    else:
        digitizer_version = "digitizer_v2"


    # DATA

    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{args.season}/station{args.station}/clean/average_ft_combined.pickle"

    data = read_freq_spectrum_from_pickle(data_path)
    frequencies = data["frequencies"]
    data_spectra = data["spectrum"]



    # SIM

    sim_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise/{digitizer_version}"
    sim_paths = [os.path.join(sim_dir, f"{component}/station{args.station}/clean/average_ft.pickle")
                 for component in ["ice", "electronic", "galactic"]]
    cross_products_path = f"{sim_dir}/cross_products/station{args.station}/cross_products.pickle"


    system_response_paths = ["sim/library/deep_templates_combined.json",
                             "sim/library/v2_v3_surface_impulse_responses.json"]


    # CALIBRATION


    calibration_dir = f"absolute_amplitude_results/season{args.season}/station{args.station}"
    calibration_names = ["measured_electronic_noise_no_weight", "cable_11", "cable_20", "cable_40"]
    calibration_paths = {cal_name : os.path.join(calibration_dir,
                                                 cal_name,
                                                 f"absolute_amplitude_calibration_season{args.season}_st{args.station}_{cal_name}_best_fit.csv")
                         for cal_name in calibration_names}
    calibrations = {cal_name : pd.read_csv(cal_path) for cal_name, cal_path in calibration_paths.items()}

    # without cable

    fitter_no_cable = spectrumFitter(data_path,
                                     sim_paths,
                                     cross_products_path,
                                     fit_range=[0.1, 0.6],
                                     remove_cable=False,
                                     include_impedance_mismatch_correction=True
                                     )

    
    fitter_cable = {length : spectrumFitter(data_path,
                                     sim_paths,
                                     cross_products_path,
                                     fit_range=[0.1, 0.6],
                                     remove_cable=True,
                                    cable_length=length,
                                     include_impedance_mismatch_correction=True
                                     )
                    for length in [11, 20, 40]}




    # PLOTTING
    plt.style.use("retro")
    
    pdf_path = f"figures/electronic_noise/compare_cable_effect_lpda_season{args.season}_st{args.station}.pdf"
    pdf = PdfPages(pdf_path)

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.plot(frequencies, data_spectra[channel_id], label = "data")

        
        fit_no_cable = fitter_no_cable.get_fit_function(mode="constant", channel_id=channel_id)
        calibrated_gain = calibrations["measured_electronic_noise_no_weight"]["gain"][channel_id]
        calibrated_template = calibrations["measured_electronic_noise_no_weight"]["best_fit_template"][channel_id]
        system_response = systemResponseTimeDomainIncorporator() 
        system_response.begin(response_path=system_response_paths, overwrite_key=calibrated_template, bandpass_kwargs=bandpass_kwargs)
        spectrum_no_cable = fit_no_cable(frequencies, calibrated_gain)
        spectrum_no_cable *= system_response.get_response(channel_id)["gain"](frequencies)
        ax.plot(frequencies, spectrum_no_cable, label="no cable")

        
        for length in [11, 20, 40]:
            fitter = fitter_cable[length]
            fit = fitter.get_fit_function(mode="constant", channel_id = channel_id)
            calibrated_gain = calibrations[f"cable_{length}"]["gain"][channel_id]
            calibrated_template = calibrations[f"cable_{length}"]["best_fit_template"][channel_id]
            system_response = systemResponseTimeDomainIncorporator() 
            system_response.begin(response_path=system_response_paths, overwrite_key=calibrated_template, bandpass_kwargs=bandpass_kwargs)
            spectrum_cable = fit(frequencies, calibrated_gain)
            spectrum_cable *= system_response.get_response(channel_id)["gain"](frequencies)
            ax.plot(frequencies, spectrum_cable, label=f"{length} m cable")




        ax.set_xlim(0., 1.)
        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("amplitude / V")
        ax.set_title(f"channel {channel_id}")
        ax.legend()

        fig.savefig(pdf, format="pdf")
        plt.close(fig)



    pdf.close()





    
