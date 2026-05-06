import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from NuRadioReco.utilities import units

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--station", type=int)
    args = parser.parse_args()
    # SETTINGS
    channels = np.arange(24)

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



    # PLOT SYSTEM RESPONSE WITH AND WITHOUT IMPEDANCE


    fitting_params_path = f"/user/rcamphyn/noise_study/absolute_amplitude_results/absolute_amplitude_calibration_season{args.season}_st{args.station}_best_fit.csv"
    fitting_params_path_no_impedance = f"/user/rcamphyn/noise_study/absolute_amplitude_results/absolute_amplitude_calibration_season{args.season}_st{args.station}_best_fit_no_impedance_mismatch.csv"


    fit = pd.read_csv(fitting_params_path)
    fit_no_impedance = pd.read_csv(fitting_params_path_no_impedance)
    
    
    
    pdf = PdfPages(f"/user/rcamphyn/noise_study/figures/impedance_mismatch/compare_without_impedance_mismatch_season{args.season}_st{args.station}.pdf")

    channels = [0, 4]
    antenna_ch_map = {ch_id : "vpol" for ch_id in [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]}
    for ch_id in [4, 8, 11, 21]:
        antenna_ch_map[ch_id] = "hpol"

    system_response_paths = ["sim/library/deep_templates_combined.json",
                             "sim/library/v2_v3_surface_impulse_responses.json"]

    for channel in channels:
        fig, ax = plt.subplots()
        template = fit["best_fit_template"][channel]
        response = systemResponseTimeDomainIncorporator()
        response.begin(system_response_paths, overwrite_key=template)

        mismatch = impedance_mismatch[antenna_ch_map[channel]]
        impedance_mismatch_gain = np.abs(mismatch)

        gain = response.get_response(channel)["gain"]
        absolute_gain = fit["gain"][channel]

        ax.plot(freqs, impedance_mismatch_gain*absolute_gain*gain(freqs), label=f"{template}\nwith impedance mismatch")
        del response


        template_no_impedance = fit_no_impedance["best_fit_template"][channel]
        response = systemResponseTimeDomainIncorporator()
        response.begin(system_response_paths, overwrite_key=template_no_impedance)

        gain = response.get_response(channel)["gain"]
        absolute_gain = fit_no_impedance["gain"][channel]

        ax.plot(freqs, absolute_gain*gain(freqs), label=f"{template}\nno impedance mismatch")

        
        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("Gain / amplitude")
        ax.legend()
        fig.suptitle(f"channel {channel}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
    

    pdf.close()
