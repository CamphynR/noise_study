import argparse
import copy
from decimal import Decimal
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pickle
from scipy import constants

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelCWNotchFilter import find_frequency_peaks, filter_cws
from NuRadioReco.utilities import units, fft

from fitting.spectrumFitter import spectrumFitter
from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, header


def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db

def convert_error_to_db(gain_error, gain):
    db_error = 20 * 1/(np.log(10) * gain) * gain_error
    return db_error


def convert_v_volt_spec_nuradio_to_temp(frequency_spectrum,
                                        resistance=50*units.ohm):

    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    temp = (frequency_spectrum)**2 / (k* resistance * 1590 * units.MHz)
    return temp



def smooth(x,window_len=11,window='hanning'):

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2)-1:-int(window_len/2)-1]

def plot_spectra_on_ax(spectrum_fitter, args, channel_id, ax, colorblind=False, x_label=True):
    idx = np.where(np.array(channel_ids) == np.array(channel_id))[0][0]
    fit_param = fit_results[idx]
    gain_factor = fit_param[0].value
    gain_error = fit_param[0].error
    el_ampl_factor = fit_param[1].value
    el_cst_factor = fit_param[2].value
    el_cst_error = fit_param[2].error
    f0_factor = fit_param[3].value

    ch_map = {"Phased array" : [0, 1, 2, 3], "HPol" : [4, 8], "LPDA up" : [13, 16, 19], "LPDA down" : [12, 14, 15, 17, 18, 20]}
    ch_map = {key : ant_type for ant_type in ch_map.keys() for key in ch_map[ant_type]}

    channel_mapping = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                       "HPol": [4, 8, 11, 21],
                       "LPDA up" : [13, 16, 19],
                       "LPDA down" : [12, 14, 15, 17, 18, 20]}

    
    fit_function = spectrum_fitter.get_fit_function(mode="electronic_temp", channel_id=channel_id)

    labels = copy.copy(noise_sources)
    # if include_sum:
    #     labels += ["sum"]
    
    if args.data is not None:
        frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(args.data)
        std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
        if channel_id in channel_mapping["LPDA up"]:
            wf = fft.freq2time(frequency_spectrum[channel_id], sampling_rate)
            filter_freqs = find_frequency_peaks(frequencies,
                                                frequency_spectrum[channel_id],
                                                threshold=2.)
            filtered_wf = filter_cws(wf, frequencies, frequency_spectrum[channel_id],
                                        sampling_rate, quality_factor=5000, threshold=2.)
            filtered_spectrum = fft.time2freq(filtered_wf, sampling_rate)
        else:
            filtered_spectrum = frequency_spectrum[channel_id]
        ax.plot(frequencies, filtered_spectrum, label = "data", lw=1.5)
        ax.fill_between(frequencies,
                        frequency_spectrum[channel_id] - std_frequency_spectrum[channel_id],
                        frequency_spectrum[channel_id] + std_frequency_spectrum[channel_id],
                        alpha=0.3)
        
    if colorblind:
        linestyles = ["dashed", "dotted", "dashdot", None]
    else:
        linestyles = [None] * 4
    for i, sim in enumerate(args.sims):
        frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(sim)
        
        frequency_spectrum = frequency_spectrum * gain_factor
        if noise_sources[i] == "electronic":
#                filt = bandpass_filt.get_filter(frequencies, -1, -1, -1, [0.1, 0.7], "butter", 10)
#                filt = np.abs(filt)
#                f0 = sampling_rate/nr_samples
            weight = el_ampl_factor*(frequencies - f0_factor) + el_cst_factor
            frequency_spectrum *= weight
#                iselect = np.nonzero(np.isclose(el_freq, frequencies, atol=0.5*f0))[0][0]
#                elec_spectral_amplitudes.append(frequency_spectrum[channel_id][iselect]/gain_factor)
#                elec_temp = convert_v_volt_spec_nuradio_to_temp(el_ampl_factor*frequency_spectrum[channel_id][iselect]/gain_factor + filt[iselect] * el_ampl_lin_factor * (frequencies[iselect] - f0_factor)
#)
            # V_rms ~ sqrt(T) => spectrum ~ sqrt(T) <=> T ~ specrum**2 (since in NuRadio spectrum is energy spectral density)
            elec_temp = 80 * units.kelvin * el_cst_factor**2
            sigma_elec_temp = 80 * 2 * el_cst_factor * el_cst_error
#                print(elec_temp)
#                elec_temp = 80
            
        if noise_sources[i] == "electronic":
            labels[i] += "\n" + r"($T_{el}|_{150 MHz}$" +  f"= {elec_temp:.2f}" + r"$\pm$" + f"{sigma_elec_temp:.2f})"
        elif noise_sources[i] == "ice":
            pass
        elif labels[i] == "sum":
            labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"

        p = ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i], lw=5., ls=linestyles[i])
        if noise_sources[i] == "electronic":
            color_electronic = p[0].get_color()

    db_error = convert_error_to_db(gain_error, gain_factor)
    ax.plot(frequencies, fit_function(frequencies, gain_factor, el_ampl_factor, el_cst_factor, f0_factor), label=f"best fit\n(gain: {convert_to_db(gain_factor):.2f}" +  r"$\pm$" +f" {db_error:.2f} dB)",
            lw=2.)

    rect_low = Rectangle((0., 0.), fit_range[0], 1., color="gray", alpha=0.5, zorder=-5)
    rect_up = Rectangle((fit_range[1], 0.), 2., 1., color="gray", alpha=0.5, zorder=-5)
    ax.add_artist(rect_low)
    ax.add_artist(rect_up)
    
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1, color="gray", lw=0.9)
    ax.legend(fontsize=24)
    ax.set_title(ch_map[channel_id], size=48)
    if x_label:
        ax.set_xlabel("frequency / GHz", size=38)
    ax.set_ylabel("spectral amplitude / V/GHz", size=38)
    ax.set_xlim(0, 1.)
    ax.set_ylim(-0.01, None)



def plot_spectra(spectrum_fitter, args, pdf, fit_results_dic, color="poster"):
    
    channel_mapping = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                       "HPol": [4, 8, 11, 21],
                       "LPDA up" : [13, 16, 19],
                       "LPDA down" : [12, 14, 15, 17, 18, 20]}

    if color=="poster":
        # darkblue, orange and lightblue
        colors = ["#011638ff", "#ee7444ff", "#e9f1f7ff"]
    else:
        colors = plt.rcParams('axes.prop_cycle')

    linestyles = [(5, (10,5)), (5,(8, 5, 1, 5)), (0, (1,5))]
    
    for idx, channel_id in enumerate(channel_ids):
        fit_param = fit_results[idx]
        print(channel_id)
        print(fit_param)
        gain_factor = fit_param[0].value
        el_ampl_factor = fit_param[1].value
        el_cst_factor = fit_param[2].value
        el_cst_error = fit_param[2].error
        f0_factor = fit_param[3].value
        gain_error = fit_param[0].error
        fit_results_dic_ch = {}
        for j in range(len(fit_param)):
            fit_results_dic_ch[fit_param[j].name] = [fit_param[j].value, fit_param[j].error]

        fit_results_dic[channel_id] = fit_results_dic_ch
        
        fit_function = spectrum_fitter.get_fit_function(mode="electronic_temp", channel_id=channel_id)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_facecolor(colors[2])
        labels = copy.copy(noise_sources)
        # if include_sum:
        #     labels += ["sum"]
        
        if args.data is not None:
            frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(args.data)
            std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
            if channel_id in channel_mapping["LPDA up"]:
                wf = fft.freq2time(frequency_spectrum[channel_id], sampling_rate)
                filter_freqs = find_frequency_peaks(frequencies,
                                                    frequency_spectrum[channel_id],
                                                    threshold=2.)
                filtered_wf = filter_cws(wf, frequencies, frequency_spectrum[channel_id],
                                            sampling_rate, quality_factor=5000, threshold=2.)
                filtered_spectrum = fft.time2freq(filtered_wf, sampling_rate)
            else:
                filtered_spectrum = frequency_spectrum[channel_id]
            ax.plot(frequencies, filtered_spectrum, label = "data", lw=2., color=colors[0])
            ax.fill_between(frequencies,
                            frequency_spectrum[channel_id] - std_frequency_spectrum[channel_id],
                            frequency_spectrum[channel_id] + std_frequency_spectrum[channel_id],
                            alpha=0.3,
                            color=colors[0])
            
        
        for i, sim in enumerate(args.sims):
            frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(sim)
            
            frequency_spectrum = frequency_spectrum * gain_factor
            if noise_sources[i] == "electronic":
#                filt = bandpass_filt.get_filter(frequencies, -1, -1, -1, [0.1, 0.7], "butter", 10)
#                filt = np.abs(filt)
#                f0 = sampling_rate/nr_samples
                weight = el_ampl_factor*(frequencies - f0_factor) + el_cst_factor
                frequency_spectrum *= weight
#                iselect = np.nonzero(np.isclose(el_freq, frequencies, atol=0.5*f0))[0][0]
#                elec_spectral_amplitudes.append(frequency_spectrum[channel_id][iselect]/gain_factor)
#                elec_temp = convert_v_volt_spec_nuradio_to_temp(el_ampl_factor*frequency_spectrum[channel_id][iselect]/gain_factor + filt[iselect] * el_ampl_lin_factor * (frequencies[iselect] - f0_factor)
#)
                # V_rms ~ sqrt(T) => spectrum ~ sqrt(T) <=> T ~ specrum**2 (since in NuRadio spectrum is energy spectral density)
                elec_temp = 80 * units.kelvin * el_cst_factor**2
                sigma_elec_temp = 80 * 2 * el_cst_factor * el_cst_error
#                print(elec_temp)
#                elec_temp = 80
                
            if noise_sources[i] == "electronic":
                labels[i] += "\n" + r"($T_{el}|_{150 MHz}$" +  f"= {elec_temp:.2f}" + r"$\pm$" + f"{sigma_elec_temp:.2f} K)"
                linewidth = 3.
            elif noise_sources[i] == "ice":
                pass
                linewidth = 3.
            elif noise_sources[i] == "galactic":
                linewidth = 5.
            elif labels[i] == "sum":
                labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"

            p = ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i], lw=linewidth, color=colors[1], ls=linestyles[i])
            if noise_sources[i] == "electronic":
                color_electronic = p[0].get_color()

        db_error = convert_error_to_db(gain_error, gain_factor)
        ax.plot(frequencies, fit_function(frequencies, gain_factor, el_ampl_factor, el_cst_factor, f0_factor), label=f"best fit \n (gain: {convert_to_db(gain_factor):.2f}" + r"$\pm$" + f"{db_error:.2f} dB)",
                lw=3., color=colors[1])

        rect_low = Rectangle((0., -0.1), fit_range[0], 1., color="gray", alpha=0.3, zorder=-5)
        rect_up = Rectangle((fit_range[1], -0.1), 2., 1., color="gray", alpha=0.3, zorder=-5)
        ax.add_artist(rect_low)
        ax.add_artist(rect_up)
        
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.1, color="gray", lw=0.9)
        ax.legend()
        ax.set_title(f"Channel {channel_id}")
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_xlim(0, 1.)
        ax.set_ylim(-0.01, None)

        smooth_window = 11 * np.diff(frequencies)[0]
        ax.text(1.01, 0.4,
                f"Fit resulst:\n gain : {gain_factor:.2f}\n electronic linear amplitude : {el_ampl_factor:.2f}\n electronic linear constant : {el_cst_factor:.2E}\n f0 : {f0_factor:.2f} GHz",
                transform=ax.transAxes,
                horizontalalignment="left",
                alpha=1., fontsize=16, bbox = {"boxstyle" : "round",
                                               "edgecolor" : "black",
                                               "facecolor" : "white"})
        fig.suptitle(f"Station {station_id}")
        fig.tight_layout()
        fig.savefig(f"figures/absolute_ampl_calibration/avg_ft_fit_s{station_id}_ch{channel_id}.svg", format="svg", dpi=600)
        fig.savefig(pdf, format="pdf")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file", default=None)
    parser.add_argument("--sims", "-s", help="path to sim pickle files, give in same order as noise sources, pass sum last ", nargs = "+")
    args = parser.parse_args()

    config_path = find_config(args.sims[0], sim=True)
    config = read_config(config_path)
    station_id = config["station"]
    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]


    bandpass_filt = channelBandPassFilter()
    

    spectrum_fitter = spectrumFitter(args.data, args.sims, bandpass=[0.1, 0.7])
    fit_results = spectrum_fitter.get_fit_gain(mode="electronic_temp")

    fit_range = spectrum_fitter.fit_range

    fit_results_dic = {ch_id : {} for ch_id in range(len(channel_ids))}

    smooth_window_len = 11
    nr_samples = 2048
    sampling_rate=3.2*units.GHz

    el_freq = 600 * units.MHz
    elec_spectral_amplitudes = []





    plt.style.use("retro")
    pdf = PdfPages(f"figures/absolute_ampl_calibration/spectra_fit_st{station_id}.pdf") 
    plot_spectra(spectrum_fitter, args, pdf, fit_results_dic)
    




    with open(f"absolute_amplitude_fits/absolute_gains_fit_season2023_s{station_id}.pickle", "wb") as pickle_file:
        pickle.dump(fit_results_dic, pickle_file)

    pdf.close()



#-----CHANNEL--SUBPLOTS------------------------------------------------------------------------------
    print("Making POS plot")
    channel_ids_plot = [0, 4, 12, 19]
    fig, axs = plt.subplots(2, 2, figsize = (40, 20), sharex=True)
    axs = np.ndarray.flatten(axs)
    x_labels = [False, False, True, True]
    for i, ax in enumerate(axs):
        plot_spectra_on_ax(spectrum_fitter, args, channel_ids_plot[i], ax, colorblind=True, x_label=x_labels[i])
        ax.text(0.99, 0.2, "Preliminary",
                color="red",
                size=42,
                alpha=1.,
                transform=ax.transAxes,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round",
                          facecolor="white",
                          edgecolor="red"))
#    fig.suptitle(f"Calibration results", size=62)
    fig.tight_layout()

    plt.savefig(f"figures/POS_ICRC/spectra_fit_st{station_id}_pos.pdf", bbox_inches="tight") 







#-----------GAINS--------------------------------------------------------------------------------
    print("Making gain per channel plot")
    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    channel_mapping = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                       "HPol": [4, 8, 11, 21],
                       "LPDA up" : [13, 16, 19],
                       "LPDA down" : [12, 14, 15, 17, 18, 20]}

    amplifier_mapping = {
            "DRAB 1" : [4, 5, 6, 7],
            "DRAB 2" : [0, 1, 2, 3],
            "DRAB 3" : [8, 9, 10, 11],
            "DRAB 4" : [21, 22, 23],
            "SURF A" : [17, 18, 19, 20],
            "SURF B" : [12, 13, 14, 15, 16]
            }


    markers = ["o", "v", "s", "D", "h", "X"]

    mapping = amplifier_mapping
    for key_i, key in enumerate(mapping.keys()):
        gain_factors = []
        el_ampl_lin_factors = []
        gain_errors = []
        el_ampl_lin_errors = []
        for idx, channel_id in enumerate(mapping[key]):
            fit_param = fit_results[channel_id]
            gain_factor = fit_param[0].value
            el_ampl_factor = fit_param[1].value
            el_ampl_lin_factor = fit_param[2].value
            f0_factor = fit_param[3].value
            gain_error = fit_param[0].error
            el_ampl_factor_error = fit_param[1].error
            el_ampl_lin_error = fit_param[2].error

            gain_factors.append(convert_to_db(gain_factor))
            gain_errors.append(convert_error_to_db(gain_error, gain_factor))
            el_ampl_lin_factors.append(el_ampl_lin_factor)
            el_ampl_lin_errors.append(el_ampl_lin_error)



        axs[0].errorbar(mapping[key], gain_factors, yerr=gain_errors, fmt=markers[key_i], label=key, markersize=10.)
        axs[1].errorbar(mapping[key], 80*np.array(el_ampl_lin_factors), yerr = 80*np.array(el_ampl_lin_errors), fmt=markers[key_i], label=key, markersize=10.)
    
    axs[0].set_title("Overal Gain")
    axs[0].legend(bbox_to_anchor=(1.,1.), loc="upper left")
    axs[1].set_ylabel("Gain / dB")
    axs[1].set_xlabel("channels")
    axs[1].set_ylabel("Eff temp @ 150 MHz / K")
    axs[1].set_title(f"Electronic amplitude at {el_freq:.2f} GHz without overal gain")
    axs[1].legend(bbox_to_anchor=(1.,1.), loc="upper left")
    axs[1].set_xticks(np.arange(24))
    fig.suptitle(f"Station {station_id}")
    fig.tight_layout()
    fig.savefig(f"figures/absolute_ampl_calibration/gain_per_channel") 
