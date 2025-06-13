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
from NuRadioReco.utilities import units

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

def plot_spectra_on_ax(spectrum_fitter, args, channel_id, ax, colorblind=False):
    idx = np.where(np.array(channel_ids) == np.array(channel_id))[0][0]
    fit_param = fit_results[idx]
    gain_factor = fit_param[0].value
    el_ampl_factor = fit_param[1].value
    el_ampl_lin_factor = fit_param[2].value
    f0_factor = fit_param[3].value
    gain_error = fit_param[0].error

    ch_map = {"Phased array" : [0, 1, 2, 3], "HPol" : [4, 8], "LPDA up" : [13, 16, 19], "LPDA down" : [12, 14, 15, 17, 18, 20]}
    ch_map = {key : ant_type for ant_type in ch_map.keys() for key in ch_map[ant_type]}

    
    fit_function = spectrum_fitter.get_fit_function(mode="electronic_temp", channel_id=channel_id)

    labels = copy.copy(noise_sources)
    # if include_sum:
    #     labels += ["sum"]
    
    if args.data is not None:
        frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(args.data)
        std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
        ax.plot(frequencies, frequency_spectrum[channel_id], label = "data", lw=1.5)
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
            weight = el_ampl_factor*(frequencies - f0_factor) + el_ampl_lin_factor
            frequency_spectrum *= weight
#                iselect = np.nonzero(np.isclose(el_freq, frequencies, atol=0.5*f0))[0][0]
#                elec_spectral_amplitudes.append(frequency_spectrum[channel_id][iselect]/gain_factor)
#                elec_temp = convert_v_volt_spec_nuradio_to_temp(el_ampl_factor*frequency_spectrum[channel_id][iselect]/gain_factor + filt[iselect] * el_ampl_lin_factor * (frequencies[iselect] - f0_factor)
#)
            elec_temp = 80 * units.kelvin * el_ampl_lin_factor
#                print(elec_temp)
#                elec_temp = 80
            
        if noise_sources[i] == "electronic":
            labels[i] += f" (T_eff_150MHz = {elec_temp:.2f} K)"
        elif noise_sources[i] == "ice":
            labels[i] += " (T = ~240 K)"
        elif labels[i] == "sum":
            labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"

        p = ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i], lw=5., ls=linestyles[i])
        if noise_sources[i] == "electronic":
            color_electronic = p[0].get_color()

    db_error = convert_error_to_db(gain_error, gain_factor)
    ax.plot(frequencies, fit_function(frequencies, gain_factor, el_ampl_factor, el_ampl_lin_factor, f0_factor), label=f"fit_result (gain: {convert_to_db(gain_factor):.2f} +- {db_error:.2f} dB)",
            lw=2.)

    rect_low = Rectangle((0., 0.), fit_range[0], 1., color="gray", alpha=0.5, zorder=-5)
    rect_up = Rectangle((fit_range[1], 0.), 2., 1., color="gray", alpha=0.5, zorder=-5)
    ax.add_artist(rect_low)
    ax.add_artist(rect_up)
    
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1, color="gray", lw=0.9)
    ax.legend()
    ax.set_title(ch_map[channel_id], size=32)
    ax.set_xlabel("freq / GHz", size=32)
    ax.set_ylabel("spectral amplitude / V/GHz", size=32)
    ax.set_xlim(0, 1.)
    ax.set_ylim(-0.01, None)



def plot_spectra(spectrum_fitter, args, pdf, fit_results_dic):
    for idx, channel_id in enumerate(channel_ids):
        fit_param = fit_results[idx]
        print(channel_id)
        print(fit_param)
        gain_factor = fit_param[0].value
        el_ampl_factor = fit_param[1].value
        el_ampl_lin_factor = fit_param[2].value
        f0_factor = fit_param[3].value
        gain_error = fit_param[0].error
        fit_results_dic_ch = {}
        for j in range(len(fit_param)):
            fit_results_dic_ch[fit_param[j].name] = [fit_param[j].value, fit_param[j].error]

        fit_results_dic[channel_id] = fit_results_dic_ch
        
        fit_function = spectrum_fitter.get_fit_function(mode="electronic_temp", channel_id=channel_id)

        fig, ax = plt.subplots(figsize=(20, 10))
        labels = copy.copy(noise_sources)
        # if include_sum:
        #     labels += ["sum"]
        
        if args.data is not None:
            frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(args.data)
            std_frequency_spectrum = np.sqrt(var_frequency_spectrum) / np.sqrt(header["nr_events"])
            ax.plot(frequencies, frequency_spectrum[channel_id], label = "data", lw=2.)
            ax.fill_between(frequencies,
                            frequency_spectrum[channel_id] - std_frequency_spectrum[channel_id],
                            frequency_spectrum[channel_id] + std_frequency_spectrum[channel_id],
                            alpha=0.3)
            
        
        for i, sim in enumerate(args.sims):
            frequencies, frequency_spectrum, var_frequency_spectrum, header = read_freq_spec_file(sim)
            
            frequency_spectrum = frequency_spectrum * gain_factor
            if noise_sources[i] == "electronic":
#                filt = bandpass_filt.get_filter(frequencies, -1, -1, -1, [0.1, 0.7], "butter", 10)
#                filt = np.abs(filt)
#                f0 = sampling_rate/nr_samples
                weight = el_ampl_factor*(frequencies - f0_factor) + el_ampl_lin_factor
                frequency_spectrum *= weight
#                iselect = np.nonzero(np.isclose(el_freq, frequencies, atol=0.5*f0))[0][0]
#                elec_spectral_amplitudes.append(frequency_spectrum[channel_id][iselect]/gain_factor)
#                elec_temp = convert_v_volt_spec_nuradio_to_temp(el_ampl_factor*frequency_spectrum[channel_id][iselect]/gain_factor + filt[iselect] * el_ampl_lin_factor * (frequencies[iselect] - f0_factor)
#)
                elec_temp = 80 * units.kelvin * el_ampl_lin_factor
#                print(elec_temp)
#                elec_temp = 80
                
            if noise_sources[i] == "electronic":
                labels[i] += f" (T_eff_150MHz = {elec_temp:.2f} K)"
            elif noise_sources[i] == "ice":
                labels[i] += " (T = ~240 K)"
            elif labels[i] == "sum":
                labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"

            p = ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i], lw=1.5)
            if noise_sources[i] == "electronic":
                color_electronic = p[0].get_color()

        db_error = convert_error_to_db(gain_error, gain_factor)
        ax.plot(frequencies, fit_function(frequencies, gain_factor, el_ampl_factor, el_ampl_lin_factor, f0_factor), label=f"fit_result (gain: {convert_to_db(gain_factor):.2f} +- {db_error:.2f} dB)",
                lw=2.)

        rect_low = Rectangle((0., 0.), fit_range[0], 1., color="gray", alpha=0.5, zorder=-5)
        rect_up = Rectangle((fit_range[1], 0.), 2., 1., color="gray", alpha=0.5, zorder=-5)
        ax.add_artist(rect_low)
        ax.add_artist(rect_up)
        
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.1, color="gray", lw=0.9)
        ax.legend()
        ax.set_title(f"Channel {channel_id}")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_xlim(0, 1.)
        ax.set_ylim(-0.01, None)

        smooth_window = 11 * np.diff(frequencies)[0]
        ax.text(1.01, 0.4,
                f"Fit resulst:\n gain : {gain_factor:.2f}\n electronic gain : {el_ampl_factor:.2f}\n electronic linear gain : {el_ampl_lin_factor:.2E}\n f0 : {f0_factor:.2f} GHz",
                transform=ax.transAxes,
                horizontalalignment="left",
                alpha=1., fontsize=16, bbox = {"boxstyle" : "round",
                                               "edgecolor" : "black",
                                               "facecolor" : "white"})
        fig.suptitle(f"Station {station_id}")
        fig.tight_layout()
        fig.savefig(f"figures/absolute_ampl_calibration/avg_ft_fit_s{station_id}_ch{channel_id}.svg", format="svg", dpi=600)
        fig.savefig(pdf, format="pdf")


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


    channel_ids_plot = [0, 4, 12, 19]
    fig, axs = plt.subplots(2, 2, figsize = (40, 20))
    axs = np.ndarray.flatten(axs)
    for i, ax in enumerate(axs):
        plot_spectra_on_ax(spectrum_fitter, args, channel_ids_plot[i], ax, colorblind=True)
    fig.suptitle(f"Station {station_id}", size=62)

    plt.savefig(f"figures/POS_ICRC/spectra_fit_st{station_id}_pos.pdf", bbox_inches="tight") 







#-----------GAINS--------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    channel_mapping = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                       "HPol": [4, 8, 11, 21],
                       "LPDA up" : [13, 16, 19],
                       "LPDA down" : [12, 14, 15, 17, 18, 20]}
    markers = ["o", "v", "s", "D", "h", "X"]

    for key_i, key in enumerate(channel_mapping.keys()):
        gain_factors = []
        el_ampl_lin_factors = []
        gain_errors = []
        el_ampl_lin_errors = []
        for idx, channel_id in enumerate(channel_mapping[key]):
            fit_param = fit_results[idx]
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


        axs[0].errorbar(channel_mapping[key], gain_factors, yerr=gain_errors, fmt=markers[key_i], label=key, markersize=10.)
        print(channel_mapping[key])
        print(len(el_ampl_lin_factors))
        axs[1].errorbar(channel_mapping[key], 80*np.array(el_ampl_lin_factors), yerr = 80*np.array(el_ampl_lin_errors), fmt=markers[key_i], label=key, markersize=10.)
    
    axs[0].set_title("Overal Gain")
    axs[0].legend()
    axs[1].set_ylabel("Gain / dB")
    axs[1].set_xlabel("channels")
    axs[1].set_ylabel("Eff temp @ 150 MHz / K")
    axs[1].set_title(f"Electronic amplitude at {el_freq:.2f} GHz without overal gain")
    axs[1].legend()
    axs[1].set_xticks(np.arange(24))
    fig.savefig(f"figures/absolute_ampl_calibration/gain_per_channel") 
