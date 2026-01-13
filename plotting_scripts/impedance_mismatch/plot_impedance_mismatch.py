import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import units


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == "__main__":
    # SETTINGS
    impedance_mismatch_correction_path = "sim/library/impedance-matching-correction-factors.npz"




    impedance_mismatch = np.load(impedance_mismatch_correction_path)

    antenna_types = ["vpol", "hpol"]

    freqs = impedance_mismatch["frequencies"]
    freq_range = [0.1, 0.7]
    mask = (freq_range[0] < freqs) & (freqs < freq_range[-1])

    plt.style.use("retro")
    fig, axs = plt.subplots(2, 2, sharex=True)
    for i, antenna_type in enumerate(antenna_types):
        mismatch = impedance_mismatch[antenna_type]
        mismatch_gain = np.abs(mismatch)
        mismatch_phase = np.angle(mismatch)
        axs[i][0].plot(freqs[mask], mismatch_gain[mask], label=antenna_type)
        axs[i][0].plot(freqs[mask], smooth(mismatch_gain[mask], 19), label=antenna_type)
        axs[i][0].set_ylabel("Gain")

        axs[i][1].plot(freqs[mask], mismatch_phase[mask], label=antenna_type)
        axs[i][1].set_ylabel("Phase")

    axs[1][0].set_xlabel("freqs / GHz")
    axs[1][1].set_xlabel("freqs / GHz")
    for ax in np.ndarray.flatten(axs):
        ax.legend()

    fig.savefig("figures/impedance_mismatch/impedance_mismatch")
