import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import os

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG











if __name__ == "__main__":
    data_dir = "/pnfs/iihe/rno-g/data/handcarry"
    station_id = 23
    run = 1657
    channel_id = 23

    run_dir = f"station{station_id}/run{run}"
    files = os.path.join(data_dir, run_dir)


    det=Detector(select_stations=station_id)
    det_time = Time("2023-08-01")
    det.update(det_time)


    reader = dataProviderRNOG()
    reader.begin(files, det=det,
                 reader_kwargs=dict(
                     select_triggers="FORCE",
                     mattak_kwargs=dict(
                        backend="uproot")))

    
    spectra = []
    for event in reader.run():
        station = event.get_station()
        channel = station.get_channel(channel_id)
        frequencies = channel.get_frequencies()
        spectrum = channel.get_frequency_spectrum()
        spectra.append(spectrum)

    spectra = np.array(spectra)
    print(len(spectra))

    avg_spectrum_gain = np.mean(np.abs(spectra), axis=0)
    avg_squared_spectrum_gain = np.sqrt(np.mean(np.abs(spectra**2), axis=0))

    plt.style.use("retro")
    fig, ax = plt.subplots()
    ax.plot(channel.get_trace())
    fig.savefig("figures/tests/trace_root.png")

    fig, ax = plt.subplots()
    ax.plot(frequencies, avg_squared_spectrum_gain)
    ax.set_xlim(0., 1.)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("spectral amplitude / V/GHz")
    ax.set_title(f"Station {station_id}, channel {channel_id}, run {run}")
    fig.savefig("figures/tests/fft_root.png")

    fig, ax = plt.subplots()
    for spec in spectra:
        ax.plot(frequencies, np.abs(spec), alpha=0.3)
    ax.set_xlim(0., 1.)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("spectral amplitude / V/GHz")
    ax.set_title(f"Station {station_id}, channel {channel_id}, run {run}")
    fig.savefig("figures/tests/fft_stacked_root.png")

        

    
