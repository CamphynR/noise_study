import sys
# temporary untill versions get reconciled on the remote
sys.path.remove("/user/rcamphyn/software/NuRadioMC")
sys.path.append("/user/rcamphyn/software/NuRadioMC_change_hardwareDB")

from astropy.time import Time
import glob
import logging
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.utilities import fft, units

from utilities.utility_functions import read_freq_spectrum_from_pickle, read_freq_spectrum_from_nur
















if __name__ == "__main__":
    
    overwrite = True
    season = 2023
    station_id = 11
    channel_id = 1
    nr_samples= 2048
    sampling_rate = 3.2 * units.GHz if season < 2024 else 2.4 * units.GHz
    frequencies = fft.freqs(nr_samples, sampling_rate)

    frequency = 300 * units.MHz
    freq_idx = np.where(np.isclose(frequency,frequencies,
                                   atol=np.diff(frequencies)[0]/2))[0][0]

    print("starting")


# -----BANDPASS-------

    bandpass = channelBandPassFilter()
    filt = bandpass.get_filter(frequencies, station_id, channel_id, det=0, passband=[0.1, 0.7], filter_type="butter", order=10)
    filt = np.abs(filt)



#------DATA--------
    hist_range = [0, 1.5]
    nr_bins = 100
    bin_width = np.diff(hist_range) / nr_bins
    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centres = bin_edges[:-1] + np.diff(bin_edges[0:2]) / 2
    spectral_histogram = np.zeros((len(frequencies), nr_bins))

    data_path = f"plotting_scripts/validation/spectra_amplitudes_season{season}_st{station_id}.csv"
    if os.path.exists(data_path) and not overwrite:
        df = pd.read_csv(data_path)
        spectra = df.to_numpy()
    else:
        spectra_paths = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/season{season}/station{station_id}/clean/spectra_run*.nur")
        spectra_paths = natsorted(spectra_paths)
        spectra = []
        for spectra_path in spectra_paths[:1]:
            spectra_tmp = read_freq_spectrum_from_nur(spectra_path)
            spectra_tmp = filt * spectra_tmp[:, channel_id, :]
            # just to avoid a concatenate at the end
            for s in spectra_tmp:
                bin_indices = np.searchsorted(bin_edges[1:-1], s)
                spectral_histogram[np.arange(len(frequencies)), bin_indices] += 1
        # normalization
        spectral_histogram = spectral_histogram / (np.sum(spectral_histogram, axis=-1)[0] * bin_width)
        df = pd.DataFrame(spectral_histogram)
        df.to_csv(data_path)


    print("done with data")






#-------SIMULATIONS------------


    ice_trace_paths = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_ice_batch*.nur")
    electronic_trace_paths = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_electronic_batch*.nur")
    galactic_trace_paths = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run*/events_galactic_batch*.nur")

    

    s_param_db_date = Time("2025-02-05", format="isot") 
    s_param_db = Detector(log_level=logging.WARNING,
                          always_query_entire_description=True,
                          select_stations=station_id,
                          database_connection="RNOG_test_public",
                          database_time=s_param_db_date,
                          database_name="RNOG_test")
    s_param_db.update(Time("2023-08-01", format="isot"))
    s_param_response = s_param_db.get_signal_chain_response(station_id, channel_id)(frequencies)


    calibration_db_date = Time("2025-08-31", format="isot") 
    calibration_db = Detector(log_level=logging.WARNING,
                              always_query_entire_description=True,
                              select_stations=station_id,
                              database_connection="RNOG_test_public",
                              database_time=calibration_db_date,
                              database_name="RNOG_test")
    calibration_db.update(Time("2023-08-01", format="isot"))
    calibration_response = calibration_db.get_signal_chain_response(station_id, channel_id)(frequencies)
#    calibration_response = np.abs(calibration_db.get_signal_chain_response(station_id, channel_id)(frequencies))


    calibration_results = pd.read_csv(f"absolute_amplitude_results/absolute_amplitude_calibration_season{season}_st{station_id}.csv")
    gains = calibration_results["gain"].to_numpy()
    el_ampl = calibration_results["el_ampl"].to_numpy()
    el_cst = calibration_results["el_cst"].to_numpy()
    f0 = calibration_results["f0"].to_numpy()

    electronic_weight = el_ampl[channel_id] * (frequencies - f0[channel_id]) + el_cst[channel_id]

    sim_spectra_cal = []
    sim_spectra_s_param = []
    for i, _ in enumerate(ice_trace_paths[:5]):
        ice_spectra = read_freq_spectrum_from_nur(ice_trace_paths[i], return_phase=True)
        electronic_spectra = read_freq_spectrum_from_nur(electronic_trace_paths[i], return_phase=True)
        galactic_spectra = read_freq_spectrum_from_nur(galactic_trace_paths[i], return_phase=True)
        
        for j, _ in enumerate(ice_spectra):
            total_sim = ice_spectra[j, channel_id] + electronic_weight * electronic_spectra[j, channel_id] + galactic_spectra[j, channel_id]
            sim_spectra_cal.append(total_sim * calibration_response)
            sim_spectra_s_param.append(total_sim * s_param_response)

    sim_spectra_cal = np.array(sim_spectra_cal)
    sim_trace_cal = fft.freq2time(sim_spectra_cal, sampling_rate)
    vrms_cal = np.sqrt(np.mean(sim_trace_cal**2, axis=-1))
    vrms_cal = np.mean(vrms_cal)

    sim_spectra_s_param = np.array(sim_spectra_s_param)
    sim_trace_s_param = fft.freq2time(sim_spectra_s_param, sampling_rate)
    vrms_s_param = np.sqrt(np.mean(sim_trace_s_param**2, axis=-1))
    vrms_s_param = np.mean(vrms_s_param)



    



#-------PLOT-----------
    plt.style.use("retro")
    fig, ax = plt.subplots()
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
#    hist, bins, patches = ax.hist(spectra[:, freq_idx], bins=20, rwidth=0.9, density=True, label="data")
    ax.bar(bin_edges[:-1], spectral_histogram[freq_idx],
           align="edge",
           fill=True,
           width=0.9*bin_width,
           label="data")
#    plt.vlines(np.mean(spectra[:, freq_idx]), 0, max(hist), lw=3., label=f"data mean@{frequency} GHz")
    ax.hist(np.abs(sim_spectra_cal[:, freq_idx]), bins=bin_edges, rwidth=0.9, density=True, label=f"calibrated sim\n(trace Vrms={vrms_cal/units.mV:.2f} mV)", alpha=0.8, zorder=10, color=colors[1])
    ax.hist(np.abs(sim_spectra_s_param[:, freq_idx]), bins=bin_edges, rwidth=0.9, density=True, label=f"s_param sim\n(trace Vrms={vrms_s_param/units.mV:.2f} mV)", alpha=0.8, zorder=15, color=colors[2])
    ax.legend()
    ax.set_xlabel("V")
    ax.set_title(f"amplitude distributions at {frequency} GHz, station {station_id}, channel {channel_id}")
    fig.savefig("figures/validation/compare_amplitude_distributions.png")
