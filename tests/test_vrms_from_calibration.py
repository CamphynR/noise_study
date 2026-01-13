import argparse
from astropy.time import Time
import copy
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np


from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.utilities import fft, units


from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from fitting.spectrumFitter import spectrumFitter
from utilities.utility_functions import read_freq_spectrum_from_pickle, read_freq_spectrum_from_nur, read_pickle









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int)
    parser.add_argument("--channel", "-c", type=int)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    
    station_id = args.station
    channel_id = args.channel
    nr_samples= 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = fft.freqs(nr_samples, sampling_rate)

# -----BANDPASS-------

    bandpass = channelBandPassFilter()
    filt = bandpass.get_filter(frequencies, station_id, channel_id, det=0, passband=[0.1, 0.7], filter_type="butter", order=10)
    filt = np.abs(filt)


#------DATA--------
    spectra_path = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/season2023/station{station_id}/clean/spectra_run*.nur")[0]
    spectra = read_freq_spectrum_from_nur(spectra_path)
    spectra = spectra[:, channel_id, :]

    vrms_data = []
    for spectrum in spectra:
        trace = fft.freq2time(spectrum, sampling_rate)
        vrms_tmp = np.sqrt(np.mean(trace**2))
        vrms_data.append(vrms_tmp)
    vrms_data = np.array(vrms_data)


    vrms_path = glob.glob(f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season2023/station{station_id}/clean/average_vrms_run*.pickle")[0]
    vrms_from_root_dic = read_pickle(vrms_path)
    vrms_from_root = vrms_from_root_dic["vrms"][channel_id]
    vrms_var_from_root = vrms_from_root_dic["var_vrms"][channel_id]


#---SIMULATIONS------
    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season2023/station{station_id}/clean/average_ft_run1930.pickle"
    sim_paths = [
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/ice/station{station_id}/clean/average_ft.pickle",
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/electronic//station{station_id}/clean/average_ft.pickle",
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/galactic/station{station_id}/clean/average_ft.pickle",
            ]

    mode = "electronic_temp"
    spectrum_fitter = spectrumFitter(
            data_path,
            sim_paths)
    fit_results = spectrum_fitter.get_fit_gain(mode=mode)
    fit_results = fit_results[channel_id]

 
    sim_trace_paths = [
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run0/events_ice_batch0.nur",
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run5/events_electronic_batch5.nur",
            f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station{station_id}/run14/events_galactic_batch14.nur",
            ]

    ice_spectra = read_freq_spectrum_from_nur(sim_trace_paths[0])
    electronic_spectra = read_freq_spectrum_from_nur(sim_trace_paths[1])
    galactic_spectra = read_freq_spectrum_from_nur(sim_trace_paths[2])

    electronic_weigth = fit_results["el_ampl"].value * (frequencies - fit_results["f0"].value) + fit_results["el_cst"].value
    electronic_spectra *= electronic_weigth



    response = systemResponseTimeDomainIncorporator()
    response.begin(det=0, response_path="sim/library/deep_impulse_responses.json")
    response = response.get_response(channel_id)["gain"](frequencies)
    response *= filt
    response *= fit_results["gain"].value

    s_param_db = Detector(log_level=logging.WARNING,
                          always_query_entire_description=True,
                          select_stations=station_id,
                          database_connection="RNOG_public")
    s_param_db.update(Time("2023-08-01", format="isot"))
    s_param_response = np.abs(s_param_db.get_signal_chain_response(station_id, channel_id)(frequencies))
    s_param_response *= filt


    ice_trace_cal = fft.freq2time(response * ice_spectra, sampling_rate)
    electronic_trace_cal = fft.freq2time(response * electronic_spectra, sampling_rate)
    galactic_trace_cal = fft.freq2time(response * galactic_spectra, sampling_rate)

    ice_trace_s = fft.freq2time(s_param_response * ice_spectra, sampling_rate)
    electronic_trace_s = fft.freq2time(s_param_response * electronic_spectra, sampling_rate)
    galactic_trace_s = fft.freq2time(s_param_response * galactic_spectra, sampling_rate)



    sim_trace_cal = ice_trace_cal[:, channel_id, :] + electronic_trace_cal[:, channel_id, :] + galactic_trace_cal[:, channel_id, :]
    sim_trace_s = ice_trace_s[:, channel_id, :] + electronic_trace_s[:, channel_id, :] + galactic_trace_s[:, channel_id, :]


    vrms_sim = []
    for trace_sim in sim_trace_cal:
        vrms_sim.append(np.sqrt(np.mean(trace_sim**2)))
    vrms_sim = np.array(vrms_sim)

            
    vrms_s_param = []
    for trace_sim in sim_trace_s:
        vrms_s_param.append(np.sqrt(np.mean(trace_sim**2)))
    vrms_s_param = np.array(vrms_s_param)



    total_sim = spectrum_fitter.get_fit_function(mode=mode, channel_id=channel_id)
    total_sim = total_sim(frequencies,
                          fit_results["gain"].value,
                          fit_results["el_ampl"].value,
                          fit_results["el_cst"].value,
                          fit_results["f0"].value)


#    test_draw = [np.array([np.random.rayleigh(scale= Vf*np.sqrt(2/np.pi)) for Vf in total_sim]) for _ in range(200)]
#    test_rms = []
#    for draw in test_draw:
#        trace = fft.freq2time(draw, sampling_rate)
#        test_rms.append(np.sqrt(np.mean(trace**2)))
#
#    test_rms = np.array(test_rms)



# -----PLOTS------

    plt.style.use("retro")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots()
    hist, bins, patches = ax.hist(vrms_data / units.mV, bins=20, rwidth=0.9, label="vrms from nur data (no bandpass)", density=True)
    ax.vlines(np.mean(vrms_data)/units.mV, 0, max(hist), linestyle="dashed", color=colors[0], lw = 2.)
    
    ax.hist(vrms_sim / units.mV, bins=20, rwidth=0.9, label="simulated vrms", density=True)
    ax.vlines(np.mean(vrms_sim) / units.mV, 0, max(hist), linestyle="dashed", color=colors[1], lw = 2.)

    ax.hist(vrms_s_param / units.mV, bins=20, rwidth=0.9, label="simulated vrms s param", density=True)
    ax.vlines(np.mean(vrms_s_param) / units.mV, 0, max(hist), linestyle="dashed", color=colors[2], lw = 2.)


    ax.vlines(vrms_from_root / units.mV, 0, max(hist), color="blue", lw=2., label="vrms from main pipeline")


    if args.log:
        ax.set_xscale("log")
    ax.set_xlabel("vrms / MV")
    ax.set_ylabel("density")
    ax.set_title(f"Station {station_id} Channel {channel_id}")
    ax.legend()
    fig.savefig("figures/tests/test_vrms_from_sim.png")



    fig, ax = plt.subplots()
    ax.plot(frequencies, filt * spectra[0], label=f"one data")
#    ax.plot(frequencies, filt * np.mean(spectra, axis=0), label=f"data, E={np.sum(filt*np.mean(spectra, axis=0)**2):.2f}")
#    ax.plot(frequencies, total_sim, label=f"sim, E={np.sum(total_sim**2):.2f}")
#    ax.plot(frequencies, np.mean(sim_from_comp, axis=0), label=f"sim from comp, E={np.sum(filt * np.mean(sim_from_comp, axis=0)**2):.2f}")
    ax.plot(frequencies, ice_spectra[0, 0], label=f"sim from comp individual")
#    ax.plot(frequencies, sim_from_s_param[0], label=f"sim from comp individual with s param, E={np.sum(sim_from_s_param[0]**2):.2f}")
#    ax.plot(frequencies, np.mean(sim_from_s_param, axis=0), label=f"sim from comp individual with s param, E={np.sum(sim_from_comp[0]**2):.2f}")
    ax.legend()
    fig.savefig("figures/tests/test_data_vs_sim_spectra")
