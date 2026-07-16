import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.interpolate import interp1d

from NuRadioReco.utilities.signal_processing import fft
from NuRadioReco.utilities import units

from utilities.utility_functions import read_freq_spectrum_from_pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int)
    args = parser.parse_args()

    station_id = args.station
    channel_ids = np.arange(24)

    nr_samples = 2048
    sampling_rate = 2.4  * units.GHz
    freqs_lower = fft.freqs(nr_samples, sampling_rate)

    sim_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/default/digitizer_v2"
    sim_comps = "ice", "electronic", "galactic"

    save_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/default/digitizer_v3_resampled"


    for sim_comp in sim_comps:
        sim = read_freq_spectrum_from_pickle(
                os.path.join(
                    sim_dir,
                    sim_comp,
                    f"station{station_id}",
                    "clean",
                    "average_ft.pickle"
            )
                )
        sim_downsample = copy.copy(sim)
        sim_downsample["frequencies"] = freqs_lower
        for channel_id in channel_ids:
            sim_interp = interp1d(sim["frequencies"], sim["spectrum"][channel_id])
            sim_downsample["spectrum"][channel_id] = sim_interp(freqs_lower)

        save_path  = os.path.join(
                save_dir,
                sim_comp,
                f"station{station_id}",
                "clean"
                    )
        os.makedirs(save_path, exist_ok=True)

        with open(f"{save_path}/average_ft.pickle", "wb") as file:
            pickle.dump(sim_downsample, file)





    cross_products_path = os.path.join(sim_dir, "cross_products", f"station{station_id}", "cross_products.pickle")
    with open(cross_products_path, "rb") as file:
        cross_products = pickle.load(file)

    cross_products_downsample = copy.copy(cross_products)
    cross_products_downsample["freq"] = freqs_lower

    for cross in ["ice_el_cross", "ice_gal_cross", "el_gal_cross"]:
        for channel_id in channel_ids:
            cross_interp_re = interp1d(cross_products["freq"], np.real(cross_products[cross][channel_id]))
            cross_interp_im = interp1d(cross_products["freq"], np.imag(cross_products[cross][channel_id]))
            cross_products_downsample[cross][channel_id] = cross_interp_re(freqs_lower) + 1j * cross_interp_im(freqs_lower)

    
    save_path  = os.path.join(
            save_dir,
            "cross_products",
            f"station{station_id}"
                )

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/cross_products.pickle", "wb") as file:
        pickle.dump(cross_products_downsample, file)

