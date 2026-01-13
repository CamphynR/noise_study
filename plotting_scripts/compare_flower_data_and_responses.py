import argparse
from astropy.time import Time
import logging
import matplotlib.backends.backend_pdf as mpl_pdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from NuRadioReco.detector.RNO_G.rnog_detector import Detector

from utilities.utility_functions import read_pickle


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, header


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to combined data pickle")
    args = parser.parse_args()
    station_id = 13

    
    data_freq, data_spectrum, data_var_spectrum, data_header = read_freq_spec_file(args.data)
    nr_channels = len(data_spectrum)

    impedance_mismatch_correction_path = "sim/library/impedance-matching-correction-factors.npz"
    impedance_mismatch_correction_npz = np.load(impedance_mismatch_correction_path)
    print(impedance_mismatch_correction_npz)
    print(impedance_mismatch_correction_npz["frequencies"])
    
    impedance_mismatch_correction = {}
    impedance_mismatch_correction["VPol"] = interp1d(impedance_mismatch_correction_npz["frequencies"], impedance_mismatch_correction_npz["vpol"], bounds_error=False, fill_value=1.) 
    impedance_mismatch_correction["HPol"] = interp1d(impedance_mismatch_correction_npz["frequencies"], impedance_mismatch_correction_npz["hpol"], bounds_error=False, fill_value=1.) 

    
    s_param_db_date = Time("2025-02-05", format="isot") 
    s_param_db = Detector(log_level=logging.WARNING,
                          always_query_entire_description=True,
                          select_stations=station_id,
                          database_connection="RNOG_test_public",
                          database_time=s_param_db_date,
                          database_name="RNOG_test")
    s_param_db.update(Time("2023-08-01", format="isot"))


    calibration_db_date = Time("2025-08-31", format="isot") 
    calibration_db = Detector(log_level=logging.WARNING,
                              always_query_entire_description=True,
                              select_stations=station_id,
                              database_connection="RNOG_test_public",
                              database_time=calibration_db_date,
                              database_name="RNOG_test")
    calibration_db.update(Time("2023-08-01", format="isot"))






    plt.style.use("retro")
    pdf = mpl_pdf.PdfPages("test.pdf")
    for channel_id in range(nr_channels):
        s_param_response = s_param_db.get_signal_chain_response(station_id, channel_id, trigger=True)
        s_param_response = np.abs(s_param_response(data_freq)) 
        s_param_response /= (np.median(s_param_response) / np.median(data_spectrum[channel_id]))
        s_param_response_corrected = s_param_response * np.abs(impedance_mismatch_correction["VPol"](data_freq))

        calibration_response = calibration_db.get_signal_chain_response(station_id, channel_id, trigger=True)
        calibration_response = np.abs(calibration_response(data_freq))
        calibration_response /= (np.median(calibration_response) / np.median(data_spectrum[channel_id]))
        calibration_response_corrected = calibration_response * np.abs(impedance_mismatch_correction["VPol"](data_freq))

        fig, ax = plt.subplots()
        ax.plot(data_freq, data_spectrum[channel_id], label="data")
        ax.plot(data_freq, s_param_response, label="s param")
        ax.plot(data_freq, s_param_response_corrected, label="s param corrected")
        ax.plot(data_freq, calibration_response, label="calibration")
        ax.plot(data_freq, calibration_response_corrected, label="calibration corrected")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude V/GHz")
        ax.legend()

        fig.savefig(pdf, format="pdf")

    pdf.close()
