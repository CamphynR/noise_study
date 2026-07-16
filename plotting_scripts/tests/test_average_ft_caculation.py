from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import units, fft

from utilities.utility_functions import read_freq_spectrum_from_root, read_freq_spectrum_from_nur, read_freq_spectrum_from_pickle












if __name__ == "__main__":
    channel_ids = np.arange(24)
    frequencies = fft.freqs(2048, 3.2) 

    path_root = "/pnfs/iihe/rno-g/data/handcarry/station11/run1956"
    path_old_spectra = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/season2023/station11/clean/spectra_run1956.nur"
    path_new_spectra = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.2/season2023/station11/clean/spectra_run1956.nur"


    spectra_root = read_freq_spectrum_from_root(path_root)
    spectra_root_mean2 = np.sqrt(np.mean(np.abs(spectra_root)**2, axis=0))

    spectra_nur_old = read_freq_spectrum_from_nur(path_old_spectra)
    spectra_nur_old_mean2 = np.sqrt(np.mean(np.abs(spectra_nur_old)**2, axis=0))

    spectra_nur_new = read_freq_spectrum_from_nur(path_new_spectra)
    spectra_nur_new_mean2 = np.sqrt(np.mean(np.abs(spectra_nur_new)**2, axis=0))

    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(6, 4, figsize=(25, 18))
    axs = np.ndarray.flatten(axs)
    for channel_id in channel_ids:
        axs[channel_id].plot(frequencies, spectra_nur_old_mean2[channel_id],
                             ls="solid", lw=1.5,
                             label="old")
        axs[channel_id].plot(frequencies, spectra_nur_new_mean2[channel_id],
                             ls="solid", lw=1.5,
                             label="new")
        axs[channel_id].plot(frequencies, spectra_root_mean2[channel_id],
                             ls="dashed", lw=2.,
                             label="root")
        axs[channel_id].set_title(f"channel {channel_id}")

    for ax in axs:
        ax.set_xlim(0., 1.)
        ax.legend(fontsize="small")

    fig_path  = "figures/tests/test_average_ft_calculations"
    fig.tight_layout()
    fig.savefig(fig_path)
