import argparse
from astropy.time import Time
import csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os

from NuRadioReco.detector.detector import Detector

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_pickle


def load_calibration_results(filename):
    with open(filename, "r") as file:
        reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            return row





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--station", type=int, default=11)
    parser.add_argument("--include_data", action="store_true")
    args = parser.parse_args()

    # DATA


    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season{args.season}/station{args.station}/clean/average_ft_combined.pickle"

    data = read_pickle(data_path)
    data_spectra = data["frequency_spectrum"]
    data_var = data["var_frequency_spectrum"]


    # S PARAM
    det = Detector(source="rnog_mongo",
                   select_stations=args.station)
    det_time = Time("2023-08-01")
    det.update(det_time)

    nr_channels = det.get_number_of_channels(args.station)
    sampling_rate = det.get_sampling_frequency(args.station)
    nr_samples = det.get_number_of_samples(args.station)

    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)



#    calibration_folder = "absolute_amplitude_results"
#    calibration_filename = os.path.join(calibration_folder,
#                                        f"absolute_amplitude_calibration_season{args.season}_st{args.station}.csv")
#    calibration = load_calibration_results(calibration_filename)


    # TIME DOMAIN TEMPLATES
    system_response = systemResponseTimeDomainIncorporator()
    if args.season == 2023:
        response_path = "sim/library/deep_impulse_responses.json"
    elif args.season == 2024:
        response_path = ["sim/library/v2_v3_deep_impulse_responses.json", "sim/library/v2_v3_surface_impulse_responses.json"]
    else:
        raise KeyError(f"{args.season} is not a recognized season.")

    system_response.begin(det=det,
                          response_path=response_path
                          )


    # PLOT
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    plt.style.use("astroparticle_physics")

    if args.include_data:
        pdf_name = f"figures/paper/compare_data_sparam_timedomain{args.season}_st{args.station}.pdf"
    else:
        pdf_name = f"figures/paper/compare_sparam_timedomain{args.season}_st{args.station}.pdf"

    pdf = PdfPages(pdf_name) 

    for channel_id in range(nr_channels):
        if args.include_data:
            s_param_response = det.get_signal_chain_response(args.station, channel_id)
            s_param_response = np.abs(s_param_response(frequencies))
            s_param_response /= np.max(s_param_response)

            calibrated_response_template = system_response.get_response(channel_id=channel_id)["gain"]

            fig, ax = plt.subplots()
            data_line = ax.plot(frequencies, data_spectra[channel_id], label="data",
                                color=list(prop_cycle)[2]["color"])
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel("data amplitude / V")
            ax.set_xlim(0, 1.)
            ax_response = ax.twinx()
            s_line = ax_response.plot(frequencies, s_param_response,
                                      label="Response from S params",
                                      ls="dashed")
            cal_line = ax_response.plot(frequencies, calibrated_response_template(frequencies),
                                        label="Calibrated response",
                                        ls="dashed")
            lines = data_line + s_line + cal_line
            labels = [l.get_label() for l in lines]
            ax_response.set_ylabel("normalized response / a.u.", rotation=-90)
            ax_response.legend(lines, labels, loc="lower left", facecolor="white")
            plt.savefig(pdf, format="pdf")
            plt.close()
        else:
            s_param_response = det.get_signal_chain_response(args.station, channel_id)
            s_param_response = np.abs(s_param_response(frequencies))
            s_param_response /= np.max(s_param_response)

            calibrated_response_template = system_response.get_response(channel_id=channel_id)["gain"]

            fig, ax = plt.subplots()
            ax.set_xlim(0, 1.)
            s_line = ax.plot(frequencies, s_param_response,
                                      label="Response from S params",
                                      ls="dashed")
            cal_line = ax.plot(frequencies, calibrated_response_template(frequencies),
                                        label="Calibrated response",
                                        ls="dashed")
            lines = s_line + cal_line
            labels = [l.get_label() for l in lines]
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel("normalized response / a.u.")
            ax.legend(lines, labels, loc="lower left", facecolor="white")
            plt.savefig(pdf, dpi=600, format="pdf")
            plt.savefig(pdf_name + f"_ch{channel_id}.png", dpi=600, format="png")
            plt.close()

        break

    pdf.close()
