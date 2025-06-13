import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import time2freq, freq2time

from modules.systemResponseTimeDomainIncorporator import systemResonseTimeDomainIncorporator



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=23)
    args = parser.parse_args()

    sampling_rate = 3.2*units.GHz
    frequencies = np.fft.rfftfreq(2048, d=1./3.2*units.GHz)

    det = Detector(source="rnog_mongo",
                   select_stations=args.station)
    det_time = Time("2023-08-01")
    det.update(det_time)


    plt.style.use("retro")


    system_response_time = systemResonseTimeDomainIncorporator()
    system_response_time.begin(det, response_path="sim/library/deep_impulse_responses.json")
    channel_ids_left = [2, 9]
    channel_ids_right = [13]
    channel_labels = {2: "deep pa component",
                      9 : "deep helper component",
                      13 : "surface component"}

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(30, 10))
    for channel_id in channel_ids_left:
        response = system_response_time.get_response(channel_id=channel_id)

        axs[0].plot(frequencies, response(frequencies), label=f"System response {channel_labels[channel_id]}",
                 lw=2.)
        axs[0].set_title("Deep antennas", size=32)

    for channel_id in channel_ids_right:
        response = system_response_time.get_response(channel_id=channel_id)

        axs[1].plot(frequencies, response(frequencies), label=f"System response {channel_labels[channel_id]}",
                 lw=2.)
        axs[1].set_title("Surface antennas", size=32)

    for ax in axs:
        ax.minorticks_on()
        ax.grid(which="minor", alpha=0.2, ls="dashed")
        ax.legend(loc=8, fontsize=26)
        ax.set_xlabel("freq / GHz", size=32)
        ax.set_xlim(0, 1.)
        ax.tick_params(axis="both", which="major", labelsize=32)
    axs[0].set_ylabel("normalized spectral amplitude / a.u.", size=32)
    fig.tight_layout()
    fig.savefig("figures/POS_ICRC/lab_measured_system_response.pdf", bbox_inches="tight")
