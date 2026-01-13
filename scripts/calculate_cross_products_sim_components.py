import argparse
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os

from NuRadioReco.utilities import fft

from utilities.utility_functions import read_freq_spectrum_from_nur, write_pickle




def combine_times(begin_time1, end_time1, begin_time2, end_time2):
    if begin_time1 < begin_time2:
        begin_time = begin_time1
    else:
        begin_time = begin_time2
    if end_time1 > end_time2:
        end_time = end_time1
    else:
        end_time = end_time2
    return begin_time, end_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=11)
    args = parser.parse_args()

    
    # SETTINGS
    station_id = args.station
    nr_samples = 2048
    sampling_rate = 3.2
    channel_ids = np.arange(24)
    test = False


    source_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.2_no_system_response_measured_electronic_noise/digitizer_v2"
    save_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.2_no_system_response_measured_electronic_noise/digitizer_v2"
    
    sim_components = [
            glob.glob(f"{source_dir}/station{station_id}/run*/events_ice_batch*.nur"),
            glob.glob(f"{source_dir}/station{station_id}/run*/events_electronic_batch*.nur"),
            glob.glob(f"{source_dir}/station{station_id}/run*/events_galactic_batch*.nur"),
            ]



    sim_components = np.array([natsorted(comp) for comp in sim_components]).T
    if test:
        sim_components = sim_components[:2]

    freqs = fft.freqs(nr_samples, sampling_rate)
    ice_el_cross = np.zeros((len(channel_ids), len(freqs)), dtype=np.complex128)
    ice_gal_cross = np.zeros((len(channel_ids), len(freqs)), dtype=np.complex128)
    el_gal_cross = np.zeros((len(channel_ids), len(freqs)), dtype=np.complex128)


    nr_events = 0
    for i, component_batch in enumerate(sim_components):
        print(i)
        ice_spectra = read_freq_spectrum_from_nur(component_batch[0], return_phase=True)
        electronic_spectra = read_freq_spectrum_from_nur(component_batch[1], return_phase=True)
        galactic_spectra = read_freq_spectrum_from_nur(component_batch[2], return_phase=True)
        nr_events += len(ice_spectra)

        for event_index, _ in enumerate(ice_spectra):
            ice_el_cross += ice_spectra[event_index] * np.conjugate(electronic_spectra[event_index]) + np.conjugate(ice_spectra[event_index]) * electronic_spectra[event_index]
            ice_gal_cross += ice_spectra[event_index] * np.conjugate(galactic_spectra[event_index]) + np.conjugate(ice_spectra[event_index]) * galactic_spectra[event_index]
            el_gal_cross += electronic_spectra[event_index] * np.conjugate(galactic_spectra[event_index]) + np.conjugate(electronic_spectra[event_index]) * galactic_spectra[event_index]


    ice_el_cross /= nr_events
    ice_gal_cross /= nr_events
    el_gal_cross /= nr_events
    
    result_dictionary = {}
    result_dictionary["freq"] = freqs
    result_dictionary["ice_el_cross"] = ice_el_cross
    result_dictionary["ice_gal_cross"] = ice_gal_cross
    result_dictionary["el_gal_cross"] = el_gal_cross
    result_dictionary["header"] = {}
    result_dictionary["header"]["nr_events"] = nr_events


    
    save_dir = f"{save_dir}/cross_products/station{station_id}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pickle_file = f"{save_dir}/cross_products.pickle"
    write_pickle(result_dictionary, pickle_file)
