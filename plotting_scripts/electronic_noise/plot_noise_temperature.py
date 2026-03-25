import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.utilities import constants, units




if __name__ == "__main__":
    
    old_dir = "/user/rcamphyn/noise_study/electronic_noise_measurements/old"
    old_files = ["electronic_noise_digitized_downhole.json",
                "electronic_noise_digitized_surface.json"]
    old_paths = [os.path.join(old_dir, filename) for filename in old_files]

    new_dir = "/user/rcamphyn/noise_study/electronic_noise_measurements/new"
    new_files = [
            "Deep_calibrated_noisetemp_-40C.csv", "Deep_calibrated_noisetemp_20C.csv",
            "Old_DRAB_calibrated_noisetemp_-40C.csv", "Old_DRAB_calibrated_noisetemp_20C.csv",
            "Old_Surf_calibrated_noisetemp_-40C.csv", "Old_Surf_calibrated_noisetemp_0C.csv", "Old_Surf_calibrated_noisetemp_20C.csv",
            "Surface_calibrated_noisetemp_-40C.csv", "Surface_calibrated_noisetemp_20C.csv"
                 ]
    new_paths = [os.path.join(new_dir, filename) for filename in new_files]


    old_index = 1
    new_index = 5

    old_path = old_paths[old_index]
    new_path = new_paths[new_index]


    old_name = old_files[old_index].split(".")[0]
    new_name = new_files[new_index].split(".")[0]


    sampling_rate = 3.2 * units.GHz
    nr_samples = 2048
    impedance = 50 * units.ohm


    old_ds = pd.read_json(old_path)
    old_ds.rename(columns={"x" : "freq", "y" : "noise_temp"}, inplace=True)
    old_ds["freq"] = old_ds["freq"] * units.MHz

    ds = pd.read_csv(new_path, names=["freq", "noise_temp"], header=None, index_col=False, skipinitialspace=True)
    ds["freq"] = ds["freq"] * units.MHz
    
    min_freq = 100 * units.MHz
    max_freq = 700 * units.MHz
    mask = (min_freq < ds["freq"]) & (ds["freq"] < max_freq)
    plt.style.use("retro")
    plt.plot(old_ds["freq"], old_ds["noise_temp"], label=old_name)
    plt.plot(ds["freq"][mask], ds["noise_temp"][mask], label=new_name)
    plt.plot()
    plt.xlim(0, 0.8)
    plt.ylim(0,300)
    plt.xlabel("freq / GHz")
    plt.ylabel("noise temperature / K")
    plt.legend()
    mean_temp = np.mean(ds["noise_temp"][mask])
    print(mean_temp)
#    plt.hlines(mean_temp, 0.1, 0.7)
    plt.savefig(f"figures/electronic_noise_measurements/noise_temp_measurements_{old_name}_vs_{new_name}")


    freqs = ds["freq"][mask]
    bandpass_filter_helper = channelBandPassFilter()
    passband = [0.15, 0.6]
    bandpass_filter = bandpass_filter_helper.get_filter(ds["freq"][mask], station_id=0, channel_id=0, det=0,
                                                passband=passband, filter_type="butter", order=10)

    df = np.diff(freqs)[0]
    power_spectral_density = constants.k_B * ds["noise_temp"][mask] * df
    power_spectral_density *= np.abs(bandpass_filter)
    energy_spectral_density = power_spectral_density/(nr_samples*sampling_rate)

    spectrum = np.sqrt(energy_spectral_density)

    spectrum =  np.sqrt(constants.k_B * ds["noise_temp"][mask] * impedance * nr_samples * sampling_rate) / sampling_rate * np.abs(bandpass_filter)

#    spectrum = np.sqrt(nr_samples * sampling_rate * constants.k_B * ds["noise_temp"][mask] * impedance) * np.abs(bandpass_filter)

    plt.clf()
    plt.plot(freqs, 700*spectrum)
    plt.savefig("test_psd_from_noise")
