from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import units, signal_processing, fft

import datetime as dt
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import interpolate
import subprocess
import sys

from utilities.utility_functions import read_pickle

# SETTINGS

plt.style.use("retro")

if len(sys.argv) > 1:
    station_ids = [int(e) for e in sys.argv[1:]]
else:
    station_ids = [11]

season = 2023

#channel_ids = [0, 1, 2, 3, 13, 16, 19]
channel_id_sets = {
        "PA" : [0, 1, 2, 3],
        "helper VPol" : [5, 6, 7, 9, 10, 22, 23],
        "HPol" : [4, 8, 11, 21],
        "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}


colors = ["blue", "red", "green", "orange", "black", "indigo", "cyan", "deeppink", "midnightblue", "brown"]


generic_noise_adder = channelGenericNoiseAdder()

nr_noise_traces = 200
nr_samples = 2048
sampling_rate = 3.2
freqs = fft.freqs(nr_samples, sampling_rate)


# DETECTOR

det = Detector(select_stations=station_ids, signal_chain_measurement_name="calibrated_impulse_response_v0")
det.update(dt.datetime(season, 6, 1))

# DATA
# load in vrms averaged over full season

print("reading pnfs")
vrms_paths = [f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season{season}/station{station_id}/clean/average_vrms_combined.pickle"
              for station_id in station_ids]
vrms_station = [read_pickle(vrms_path) for vrms_path in vrms_paths]

vrms_hist_paths = [f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms_histograms/complete_vrms_histograms_sets/season2023/station{station_id}/clean/vrms_histograms_combined.pickle"
          for station_id in station_ids]
vrms_hist_station = [read_pickle(vrms_hist_path) for vrms_hist_path in vrms_hist_paths]
print("done reading pnfs")


# SAVE DIRECTORY

save_dir = "/user/rcamphyn/noise_study/effective_temperature"


# INTERSECT AND PLOT

temperatures = np.linspace(200, 600, 10) # Kelvin
for station_index, station_id in enumerate(station_ids):
    # data
    header = vrms_station[station_index]["header"]
    data_vrms_average = vrms_station[station_index]["vrms"]
    data_vrms_var = vrms_station[station_index]["var_vrms"]

    # simulation
    vrms_min=1e3
    vrms_max = 0

    # validation

    header_vrms_hist = vrms_hist_station[station_index]["header"]
    bin_centres = np.array(header_vrms_hist["bin_centres"])
    bin_width = np.diff(bin_centres)[0]
    bin_edges = bin_centres - 0.5*bin_width
    bin_edges = bin_edges.tolist()
    bin_edges.append(bin_centres[-1] + 0.5 * bin_width)
    data_vrms_hist = np.array(vrms_hist_station[station_index]["vrms_hist"])

    figsize = (30, 20)
    ax_nr_columns = 2
    ax_nr_rows = int(len(channel_id_sets)/ax_nr_columns)
    ax_shape = (ax_nr_rows, ax_nr_columns)
    fig, axs = plt.subplots(*ax_shape, figsize=figsize)
    axs = np.ndarray.flatten(axs)
    fig_validation, axs_validation = plt.subplots(*ax_shape, figsize=figsize)
    axs_validation = np.ndarray.flatten(axs_validation)  

    eff_temps = {}
    for channel_set_index, (channel_set_name, channel_ids) in enumerate(channel_id_sets.items()):
        print(channel_set_name)
        ax = axs[channel_set_index]
        ax_validation = axs_validation[channel_set_index]
        for channel_index, channel_id in enumerate(channel_ids):
            resp = det.get_signal_chain_response(station_id, channel_id)
            vrms = signal_processing.calculate_vrms_from_temperature(temperatures, response=resp)
            ax.hlines(data_vrms_average[channel_id] / units.mV, temperatures[0], temperatures[-1],
                        color=colors[channel_index], ls="dashed")

            # find intersection by drawing straight line through two closest points
            intersection_index = np.searchsorted(vrms, data_vrms_average[channel_id])
            rico = (vrms[intersection_index] - vrms[intersection_index - 1]) / (temperatures[intersection_index] - temperatures[intersection_index - 1])
            eff_temp = (data_vrms_average[channel_id] - vrms[intersection_index - 1]) * rico**-1 + temperatures[intersection_index - 1]
            ax.vlines(eff_temp, 0, 15,
                        color=colors[channel_index], ls="dashed")
            ax.plot(temperatures, vrms / units.mV, color=colors[channel_index], linewidth=1,
                    label=f"channel {channel_id},\nT_eff={eff_temp:.2f} K")
            if max(vrms) > vrms_max:
                vrms_max = max(vrms)
            if min(vrms) < vrms_min:
                vrms_min = min(vrms)
            
            eff_temps[channel_id] = eff_temp


                
            # validation
            vrms_from_eff_temp = signal_processing.calculate_vrms_from_temperature(eff_temp, response=resp) 
            
            noise_vrms = []
            for _ in range(nr_noise_traces):
                noise = generic_noise_adder.bandlimited_noise(0.1, 0.7, nr_samples, sampling_rate,
                                                                amplitude=vrms_from_eff_temp,
                                                                type="rayleigh",
                                                                time_domain=True)
                noise_vrms_tmp = np.sqrt(np.mean(noise**2))
                noise_vrms.append(noise_vrms_tmp)
            

            # normalize
            normalized_data_vrms_hist = data_vrms_hist[channel_id] / np.max(data_vrms_hist[channel_id])
            ax_validation.bar(bin_centres / units.mV, normalized_data_vrms_hist, width=bin_width / units.mV, 
                    edgecolor=colors[channel_index],
                    facecolor="white",
                    label=f"channel {channel_id} data") 
        
            simulated_vrms_hist, _ = np.histogram(noise_vrms, bin_edges)
            simulated_vrms_hist = simulated_vrms_hist / np.max(simulated_vrms_hist)
            ax_validation.bar(bin_centres / units.mV, simulated_vrms_hist, width=bin_width / units.mV, 
                    edgecolor=colors[channel_index],
                    facecolor=colors[channel_index],
                    ls="dashed",
                    label=f"channel {channel_id} sim") 
    #        ax_validation.hist(noise_vrms, bins=bin_edges/units.mV, color=colors[channel_id],
                    
    #                label=f"channel {channel_id} sim") 

            
        
        

        
        ax.set_ylim(0.95 * vrms_min / units.mV, 1.05 * vrms_max / units.mV)
        if channel_set_index % ax_nr_columns == 0:
            legend_loc = "upper right"
            bbox_to_anchor = (0., 1.)
        else:
            legend_loc = "upper left"
            bbox_to_anchor = (1., 1.)
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("VRMS (mV)")
        ax.set_title(channel_set_name)

        ax_validation.legend(fontsize=8)
        ax_validation.set_xlabel("Vrms / mV")
        ax_validation.set_ylabel("normalized counts")
        ax_validation.set_title(channel_set_name)

    fig.tight_layout()
    fig.savefig(f"figures/effective_temperatures/eff_temperature_season{season}_st{station_id}.png")
    fig_validation.savefig(f"figures/effective_temperatures/eff_temperature_validation_season{season}_st{station_id}.png")


    filename = f"eff_temperatures_season{season}_station{station_id}.json"
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w") as file:
        json.dump(eff_temps, file)
