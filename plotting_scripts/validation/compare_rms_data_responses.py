import sys
# temporary untill versions get reconciled on the remote
sys.path.remove("/user/rcamphyn/software/NuRadioMC")
sys.path.append("/user/rcamphyn/software/NuRadioMC_change_hardwareDB")


import argparse
from astropy.time import Time
import copy
import glob
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from scipy import constants

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelThermalNoiseAdder import channelThermalNoiseAdder
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units, fft

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from utilities.utility_functions import read_pickle, read_config, read_freq_spectrum_from_nur



def temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance=50*units.ohm, filter_type="rectangular"):
    if filter_type=="rectangular":
        print(min_freq)
        print(frequencies)
        filt = np.zeros_like(frequencies)
        filt[np.where(np.logical_and(min_freq < frequencies , frequencies < max_freq))] = 1
    else:
        print("Other filters not yet implemented")
    bandwidth = np.trapz(np.abs(filt)**2, frequencies)
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    vrms = np.sqrt(k * temperature * resistance * bandwidth)
    return vrms


def generate_event(station_id, frequencies, channel_ids):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time("2023-08-01", format="isot")
    for channel_id in channel_ids:
        channel = Channel(channel_id)
        channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
        station.add_channel(channel)
    event.set_station(station)
    return event


def initialize_ice_adder():
    eff_temp_dir = "sim/library/eff_temperatures"
    ice_adder = channelThermalNoiseAdder()
    ice_adder.begin(sim_library_dir=eff_temp_dir)
    return ice_adder


def initialize_electronic_adder(min_freq, max_freq, frequencies, electronic_temp):
    resistance = 50 * units.ohm
    amplitude = temp_to_volt(electronic_temp, min_freq, max_freq, frequencies, resistance, filter_type="rectangular")
    electronic_adder = channelGenericNoiseAdder()

    return electronic_adder, amplitude


def initialize_galactic_adder(min_freq, max_freq):
    galactic_adder = channelGalacticNoiseAdder()
    galactic_adder.begin(freq_range=[min_freq, max_freq], caching=True)

    return galactic_adder


def add3components(event, station, det,
                   min_freq=10*units.MHz,
                   max_freq=1500*units.MHz,
                   electronic_temp=80*units.K):
    freqs = event.get_station().get_channel(0).get_frequencies()
    print(freqs.shape)
    ice_adder = initialize_ice_adder()
    electronic_adder, electronic_amp = initialize_electronic_adder(min_freq, max_freq, freqs, electronic_temp)
    galactic_adder = initialize_galactic_adder(min_freq, max_freq)

    ice_adder.run(event, station, det)
    electronic_adder.run(event, station, det, electronic_amp)
    galactic_adder.run(event, station, det)



def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return {"frequencies" : frequencies,
            "spectrum" : frequency_spectrum,
            "var_spectrum" : var_frequency_spectrum,
            "header" : header}



#def read_freq_spectrum_from_nur(files : list):
#    event_reader = eventReader()
#    event_reader.begin(files)
#    spec = []
#    for event in event_reader.run():
#        station = event.get_station()
#        spec_channel = []
#        for channel in station.iter_channels():
#            frequency_spectrum = channel.get_frequency_spectrum()
#            spec_channel.append(np.abs(frequency_spectrum))
#        spec.append(spec_channel)
#    return np.array(spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int)
    args = parser.parse_args()

    config_path = "sim/thermal_noise/config_efields.json"
    config = read_config(config_path)

    season = 2023
    station_id = args.station
    nr_channels = 24
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    channel_ids = np.arange(24)
#    channel_ids = [5]
    frequencies = fft.freqs(nr_samples, sampling_rate) 







# -----BANDPASS-------

    bandpass = channelBandPassFilter()
    filt = bandpass.get_filter(frequencies, station_id=0, channel_id=0, det=0, passband=[0.1, 0.7], filter_type="butter", order=10)
    filt = np.abs(filt)


# ----------SIMULATION--------------

    s_param_db_date = Time("2025-02-05", format="isot") 
    s_param_db = Detector(log_level=logging.WARNING,
                          always_query_entire_description=True,
                          select_stations=station_id,
                          database_connection="RNOG_test_public",
                          database_time=s_param_db_date,
                          database_name="RNOG_test")
    s_param_db.update(Time("2023-08-01", format="isot"))
#    s_param_response = np.array(
#            [np.abs(s_param_db.get_signal_chain_response(station_id, c)(frequencies)) for c in channel_ids]
#            )

    s_param_response = np.array(
            [s_param_db.get_signal_chain_response(station_id, c)(frequencies) for c in channel_ids]
            )

    calibration_db_date = Time("2025-08-31", format="isot") 
    calibration_db = Detector(log_level=logging.WARNING,
                              always_query_entire_description=True,
                              select_stations=station_id,
                              database_connection="RNOG_test_public",
                              database_time=calibration_db_date,
                              database_name="RNOG_test")
    calibration_db.update(Time("2023-08-01", format="isot"))
#    calibration_response = np.array(
#            [np.abs(calibration_db.get_signal_chain_response(station_id, c)(frequencies)) for c in channel_ids]
#            )

    calibration_response = np.array(
            [calibration_db.get_signal_chain_response(station_id, c)(frequencies) for c in channel_ids]
            )


    radiant_calibration_path = f"absolute_amplitude_results/absolute_amplitude_calibration_season{season}_st{station_id}.csv"
    calibration_parameters = pd.read_csv(radiant_calibration_path)
    el_ampl = calibration_parameters["el_ampl"].to_numpy()
    el_cst = calibration_parameters["el_cst"].to_numpy()
    f0 = calibration_parameters["f0"].to_numpy()

    electronic_weight_frequencies = np.tile(frequencies, (24, 1))
    el_ampl = np.tile(el_ampl, (len(frequencies),1)).T
    el_cst = np.tile(el_cst, (len(frequencies),1)).T
    f0 = np.tile(f0, (len(frequencies),1)).T
    electronic_weight = el_ampl * (electronic_weight_frequencies - f0) + el_cst



    thermal_noise_trace_paths = [
                                glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_ice_batch*.nur"),
                                glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_electronic_batch*.nur"),
                                glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_galactic_batch*.nur")
            ]
    thermal_noise_trace_paths = np.array(thermal_noise_trace_paths).T

    vrms_s_param = []
    vrms_calibration = []
    for event_batch in thermal_noise_trace_paths[:1]:
        spec_per_batch = []
        # 0 ice 1 electronic 2 galactic
        for i, component in enumerate(event_batch):
            spec = read_freq_spectrum_from_nur(component, return_phase=True)
            if i==1:
                # electronic noise
                spec *= electronic_weight

            spec_per_batch.append(spec)
       
        
#        trace_s = np.sum(fft.freq2time(s_param_response * spec_per_batch, sampling_rate), axis=0)
#        trace_cal = np.sum(fft.freq2time(calibration_response * spec_per_batch, sampling_rate), axis=0)
        trace_s = fft.freq2time(np.sum(s_param_response * spec_per_batch, axis=0), sampling_rate)
        trace_cal = fft.freq2time(np.sum(calibration_response * spec_per_batch, axis=0), sampling_rate)


        vrms_s_tmp = np.sqrt(np.mean(trace_s**2, axis=-1))
        vrms_s_param.append(vrms_s_tmp)

        vrms_cal_tmp = np.sqrt(np.mean(trace_cal**2, axis=-1))
        vrms_calibration.append(vrms_cal_tmp)


    vrms_s_param = np.concatenate(vrms_s_param, axis=0)
    vrms_s_param = np.mean(vrms_s_param, axis=0)
    vrms_calibration = np.concatenate(vrms_calibration, axis=0)
    vrms_calibration = np.mean(vrms_calibration, axis=0)
    
 
#----------GAINS PER RUN-------------

    gains_per_run_paths = natsorted(glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/system_amplitude_calibration/season{season}/station{station_id}/results_per_run/absolute_amplitude_calibration_season{season}_st{station_id}_run*.csv"))
    gains_per_run_paths = [g for g in gains_per_run_paths if not g.split(".csv")[0].endswith("error")]
    response_template_path = "sim/library/deep_impulse_responses.json"
    response_template = systemResponseTimeDomainIncorporator()
    response_template.begin(0, response_template_path)
    response_template = np.array([response_template.get_response(c)["gain"](frequencies) for c in channel_ids])


    n_gain_runs = 30
    d_gain = int(len(gains_per_run_paths)/n_gain_runs)
    tmp_idx=0

    vrms_per_run = []
    for gain_run_path in gains_per_run_paths[::d_gain]:
        with open(gain_run_path, "r") as gain_file:
            gain_run = pd.read_csv(gain_file)["gain"].to_numpy()
        vrms_run = []
        print(f"gain_run {tmp_idx}")
        tmp_idx+=1
        for event_batch in thermal_noise_trace_paths[:1]:
            spec_per_batch = []
            # 0 ice 1 electronic 2 galactic
            for i, component in enumerate(event_batch):
                spec = read_freq_spectrum_from_nur(component)
                if i==1:
                    # electronic noise
                    spec *= electronic_weight

                spec_per_batch.append(spec)
           
            
            trace_run = np.sum(fft.freq2time(np.multiply(gain_run[:, np.newaxis], response_template) * spec_per_batch, sampling_rate), axis=0)


            vrms_run_tmp = np.sqrt(np.mean(trace_run**2, axis=-1))
            vrms_run.append(vrms_run_tmp)
        vrms_run = np.concatenate(vrms_run, axis=0)
        vrms_run = np.mean(vrms_run, axis=0)
        vrms_per_run.append(vrms_run)
    vrms_per_run = np.array(vrms_per_run).T







#------------DATA-------------


    data_paths = natsorted(glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season{season}/station{station_id}/clean/average_vrms_run*.pickle"))

    times = []
    vrms = []
    var_vrms = []
    for pickle in data_paths:
        rms_dict = read_pickle(pickle)
        times.append(rms_dict["header"]["begin_time"].unix)
        vrms.append(rms_dict["vrms"])
        var_vrms.append(rms_dict["var_vrms"])
    vrms = np.array(vrms).T
    var_vrms = np.array(var_vrms).T

    times = np.array(times)[:, 0]
    times_date = [Time(t, format="unix").strftime("%Y-%B-%d") for t in times]


    

#----------------PLOTS------------------


    pdf = PdfPages(f"figures/validation/compare_rms_system_responses_season{season}_st{station_id}.pdf")
    for channel_id in channel_ids:
        plt.style.use("retro")
        fig, ax = plt.subplots()
        ax.errorbar(times, vrms[channel_id]/units.mV, yerr=var_vrms[channel_id]/units.mV, marker="o", label="Vrms", ls=None)
        ax.plot(times[::d_gain], vrms_per_run[channel_id]/units.mV, marker="v", label="Calibrated per run", lw=1., color="pink")
        ax.hlines(vrms_calibration[channel_id]/units.mV, xmin=times[0], xmax=times[-1], label = "calibrated response", colors="red")
        ax.hlines(vrms_s_param[channel_id]/units.mV, xmin=times[0], xmax=times[-1], label = "combined S parameter response", colors="blue")
        ax.set_xlabel("date")
        ax.set_ylabel(r"$V_{rms}$ / mV")
        dt = int(len(times)/10)
        ax.set_xticks(times[::dt], labels=times_date[::dt], rotation=-45, ha="left", rotation_mode="anchor")
        ax.set_title(f"Station {station_id}, channel {channel_id}")
        ax.minorticks_on()
        ax.grid(True, which="minor", ls="dashed", color="gray", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()
