"""
author: Felix
modified by Ruben
"""

import argparse
from astropy.time import Time
import itertools
from matplotlib import pyplot as plt
import numpy as np

from NuRadioReco.detector import antennapattern
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=23)
    args = parser.parse_args()


    antenna_provider = antennapattern.AntennaPatternProvider()
    azimuth = np.linspace(0, 2 * np.pi)
    zenith = np.linspace(0, 89 / 180 * np.pi)

    det = detector.Detector(source="rnog_mongo", select_stations=args.station)
    detector_time = Time("2022-08-01")
    det.update(detector_time)
    station_info = det.get_station(args.station)

    frequencies = np.fft.rfftfreq(station_info["number_of_samples"], d=1./station_info["sampling_rate"])

    antenna_models = ["RNOG_vpol_ice_upsample", "RNOG_hpol_v4_8inch_center_n1.74"]

    plot_freq = 0.5
    vels = []
    freqs = np.around(np.arange(0.01, 1, 0.0001), 4)
    mask = freqs == plot_freq


    assert np.any(mask), "No frequency selected"

    for channel_id, antenna_model in zip([0, 1], antenna_models):


        ori = [0, 0, np.pi / 2, np.pi / 2]
        antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model)
        vel = []
        for azi, zen in itertools.product(azimuth, zenith):
            VEL = antenna_pattern.get_antenna_response_vectorized(freqs, zen, azi, *ori)
            vel.append([np.abs(VEL["theta"]), np.abs(VEL["phi"])])

        vels.append(vel)


    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, gridspec_kw=dict(top=0.95, wspace=0.25, bottom=0.05, left=0.02, right=0.99), figsize=(7, 4))
    vels = np.array(vels)

    vmin = np.amin(vels[:, :, :, mask])
    vmax = np.amax(vels[:, :, :, mask])
    for idx, vel in enumerate(vels):
        vel = vel[:, :, mask].reshape(len(azimuth), len(zenith), 2).T
        ant = "Vpol" if idx == 0 else "HPol"
        if idx == 0:
            axs[idx].set_title(f"Vpol (theta)")
            axs[idx].pcolormesh(azimuth, np.rad2deg(zenith), vel[0], shading='gouraud', vmin=vmin, vmax=vmax)
        else:
            axs[idx].set_title(f"Hpol (phi)")
            pcm = axs[idx].pcolormesh(azimuth, np.rad2deg(zenith), vel[1], shading='gouraud', vmin=vmin, vmax=vmax)

    cb = plt.colorbar(pcm, ax=axs.ravel().tolist(), label="VEL [m]")

    fig.suptitle(f"Antenna pattern at {plot_freq} GHz")

    # fig.tight_layout()
    fig.savefig("antenna_rnog.png")

    for channel_id, antenna_model in zip([0, 1], antenna_models):

        fig, ax = plt.subplots()

        zenith = 50 * units.deg
        azimuth = 20 * units.deg

        # vel = vels[0].reshape(len(azimuth), len(zenith), len(freqs), 2)[13, 25, :, 1]
        antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model)
        pol = "theta" if channel_id == 0 else "phi"
        vel = np.abs(antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth, *[0, 0, np.pi / 2, np.pi / 2])[pol])

        ax.plot(freqs, vel)
        print(azimuth / units.deg)
        ax.set_title(f"{antenna_model}, zenith={zenith/units.deg:.1f}, azimuth={azimuth/units.deg:.1f}")
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("VEL / m")
        # ax.set_xlim(0.04, 0.07)
        # ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(f"{antenna_model}_antenna_freq_log.png")
        print(vel.shape)
