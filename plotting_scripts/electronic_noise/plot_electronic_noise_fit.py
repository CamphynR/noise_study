import argparse
import copy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


from NuRadioReco.utilities import units



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--station", type=int)
    args = parser.parse_args()

    # ELECTTRONIC TEMPERATURE MEASUREMENTS

    electronic_noise_measurements_paths = \
            ["/user/rcamphyn/noise_study/RF_chain_testing/october_drabsurf_freezer/Calibrated_noise_figures/Old_DRAB_calibrated_noisetemp_-40C.csv",
             "/user/rcamphyn/noise_study/electronic_noise_measurements/new/Old_Surf_calibrated_noisetemp_0C.csv"]

    electronic_measurements = {}
    electronic_measurements["drab"] = pd.read_csv(electronic_noise_measurements_paths[0], names=["freq", "temp"], header=None, index_col=False, skipinitialspace=True)
    electronic_measurements["surface"] = pd.read_csv(electronic_noise_measurements_paths[1], names=["freq", "temp"], header=None, index_col=False, skipinitialspace=True)



    # SETTINGS
    fit_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{args.season}/station{args.station}/default"
    fit_filename = f"absolute_amplitude_calibration_season{args.season}_st{args.station}_best_fit.csv"
    fit_path = os.path.join(fit_dir, fit_filename)
    
    fit_results = pd.read_csv(fit_path)

    channels = fit_results.iloc[:, 0]
    electronic_slope = fit_results["el_ampl"].to_numpy()
    electronic_offset = fit_results["f0"].to_numpy()


    # PLOT THE FIT VALUES
    plt.style.use("retro")
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(channels, electronic_slope)
    axs[0].set_ylabel("electronic slope")
    axs[1].scatter(channels, electronic_offset)
    axs[1].set_xlabel("frequencies / GHz")
    axs[1].set_ylabel("electronic offset")
    axs[1].set_xticks(channels)
    fig.suptitle(f"electronic noise fit parameters season {args.season} station {args.station}")
    fig.savefig(f"figures/electronic_noise_fit_values_season{args.season}_st{args.station}")


    # CONVERT FIT VALUES TO PERCENTAGE DIFFERENCE ON ELECTTRONIC SPECTRUM
    # function used on spectrum is S' = [el_slope (f - el_offset) + 1] * S
    # see that at f = el_offset the spectrum remains unchanged
    # Note fit was performed in GHz (since el_slope has dim 1/GHz)


    def electronic_spectrum_weight_function(f, el_slope, el_offset):
        return el_slope[:, np.newaxis] * (f - el_offset[:, np.newaxis]) + 1

    frequencies = np.linspace(0, 1., 1000)
    frequencies = np.tile(frequencies, (len(channels), 1))
    electronic_spectrum_weight = electronic_spectrum_weight_function(frequencies, electronic_slope, electronic_offset)
    temperature_difference = (1 - electronic_spectrum_weight**2) * 100


    subplot_channels = [[0, 1, 2, 3], [4, 8, 11, 21], [5, 6, 7, 9, 10, 22, 23], [12, 13, 14, 15, 16, 17, 18, 19, 20]]

    subplot_names = ["PA", "HPols", "helper VPols", "LPDA"]
    pdf = PdfPages(f"figures/electronic_noise_fit_temp_difference_season{args.season}_st{args.station}.pdf")
    for pdf_idx in range(len(subplot_channels)):
        fig, ax = plt.subplots()
        for ch_idx in subplot_channels[pdf_idx]:
            ax.plot(frequencies[ch_idx], temperature_difference[ch_idx], label = f"channel {ch_idx}")
        
        ax.set_xlim(0.1, 0.6)
        ax.set_ylim(-20, 20)
        ax.legend(ncol=2, bbox_to_anchor=(1., 1.), loc="upper left")
        fig.tight_layout()
        fig.suptitle(subplot_names[pdf_idx])
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()



    # PLOT EFFECT IN TEMP

    
    electronic_spectrum_weight_drab = electronic_spectrum_weight_function(np.tile(electronic_measurements["drab"]["freq"] * units.MHz, (len(channels), 1)),
                                                                          electronic_slope, electronic_offset)
    electronic_spectrum_weight_surface = electronic_spectrum_weight_function(np.tile(electronic_measurements["surface"]["freq"] * units.MHz, (len(channels), 1)),
                                                                             electronic_slope, electronic_offset)
    electronic_spectrum_weight_per_component = {}
    electronic_spectrum_weight_per_component["drab"] = electronic_spectrum_weight_drab
    electronic_spectrum_weight_per_component["surface"] = electronic_spectrum_weight_surface



    surface_channels = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    freq_range = [0.1, 0.6]
    pdf = PdfPages(f"figures/electronic_noise_fit_fitted_temp_season{args.season}_st{args.station}.pdf")
    for pdf_idx in range(len(subplot_channels)):
        fig, ax = plt.subplots()
        for i, ch_idx in enumerate(subplot_channels[pdf_idx]):
            if ch_idx in surface_channels:
                key = "surface"
            else:
                key = "drab"
            frequencies_measurement = electronic_measurements[key]["freq"] * units.MHz
            freq_idx = (freq_range[0] < frequencies_measurement) & (frequencies_measurement < freq_range[1])
            frequencies_measurement = frequencies_measurement[freq_idx]
            temp_measurement = copy.deepcopy(electronic_measurements[key]["temp"])
            temp_measurement *= electronic_spectrum_weight_per_component[key][ch_idx]**2
            if i == 0:
                ax.plot(frequencies_measurement, electronic_measurements[key]["temp"][freq_idx], label="data", lw=4.)
            ax.plot(frequencies_measurement, temp_measurement[freq_idx], label=f"channel {ch_idx}")

#        ax.set_xlim(0.1, 0.6)
        ax.legend(ncol=2, bbox_to_anchor=(1., 1.), loc="upper left")
        fig.suptitle(subplot_names[pdf_idx])
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()


    # different layout
    subplot_channels = [[0, 1, 2, 3, 9, 10, 22, 23, 5, 6, 7], [4, 8, 11, 21], [5, 6, 7], [12, 13, 14, 15, 16, 17, 18, 19, 20]]

    subplot_names = ["deep VPOLS", "HPols", "shallow VPols", "LPDA"]
    pdf = PdfPages(f"figures/electronic_noise_fit_fitted_temp_season{args.season}_st{args.station}_alt.pdf")
    for pdf_idx in range(len(subplot_channels)):
        fig, ax = plt.subplots()
        for i, ch_idx in enumerate(subplot_channels[pdf_idx]):
            if ch_idx in surface_channels:
                key = "surface"
            else:
                key = "drab"
            frequencies_measurement = electronic_measurements[key]["freq"] * units.MHz
            freq_idx = (freq_range[0] < frequencies_measurement) & (frequencies_measurement < freq_range[1])
            frequencies_measurement = frequencies_measurement[freq_idx]
            temp_measurement = copy.deepcopy(electronic_measurements[key]["temp"])
            temp_measurement *= electronic_spectrum_weight_per_component[key][ch_idx]**2
            if i == 0:
                ax.plot(frequencies_measurement, electronic_measurements[key]["temp"][freq_idx], label="data", lw=4.)
            if pdf_idx == 0 and ch_idx in [5, 6, 7]:
                alpha = 0.5
                lw=0.5
            else:
                alpha = 1.
                lw=2.
            ax.plot(frequencies_measurement, temp_measurement[freq_idx], label=f"channel {ch_idx}",alpha=alpha)

#        ax.set_xlim(0.1, 0.6)
        ax.legend(ncol=2, bbox_to_anchor=(1., 1.), loc="upper left")
        fig.suptitle(subplot_names[pdf_idx])
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
