import argparse
from astropy.time import Time
import csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os

from NuRadioReco.detector.detector import Detector

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator


def load_calibration_results(filename):
    with open(filename, "r") as file:
        reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            return row





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--station", type=int, default=11)
    args = parser.parse_args()



    det = Detector(source="rnog_mongo",
                   select_stations=args.station)
    det_time = Time("2023-08-01")
    det.update(det_time)

    nr_channels = det.get_number_of_channels(args.station)
    sampling_rate = det.get_sampling_frequency(args.station)
    nr_samples = det.get_number_of_samples(args.station)

    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)



    calibration_folder = "absolute_amplitude_results"
    calibration_filename = os.path.join(calibration_folder,
                                        f"absolute_amplitude_calibration_season{args.season}_st{args.station}.csv")
    calibration = load_calibration_results(calibration_filename)

    system_response = systemResonseTimeDomainIncorporator()
    if args.season == 2023:
        response_path = "sim/library/deep_impulse_responses.json"
    elif args.season == 2024:
        response_path = ["sim/library/v2_v3_deep_impulse_responses.json", "sim/library/v2_v3_surface_impulse_responses.json"]
    else:
        raise KeyError(f"{args.season} is not a recognized season.")

    system_response.begin(det=det,
                          response_path=response_path
                          )

    plt.style.use("retro")
    pdf = PdfPages(f"figures/response_comparisons/compare_s_vs_calibration_season{args.season}_st{args.station}.pdf") 
    for channel_id in range(nr_channels):
        calibrated_gain = calibration[channel_id]
        calibrated_response = system_response.get_response(channel_id=channel_id)
        calibrated_response = calibrated_gain * np.abs(calibrated_response(frequencies))
        
        s_param_response = det.get_signal_chain_response(args.station, channel_id)
        s_param_response = np.abs(s_param_response(frequencies))

        fig, ax = plt.subplots()
        ax.plot(frequencies, s_param_response, label="Response from S params")
        ax.plot(frequencies, calibrated_response, label="Calibrated response")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("Gain / amplitude")
        ax.set_title(f"Season {args.season} station {args.station} channel {channel_id}")
        ax.legend()
        plt.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()
