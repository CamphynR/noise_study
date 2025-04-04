import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter

from fitting.spectrumFitter import spectrumFitter
from utilities.utility_functions import read_pickle, find_config, read_config


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    return frequencies, frequency_spectrum


def convert_to_db(gain):
    db = 20*np.log10(gain)
    return db


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="path to dat pickle file", default=None)
    parser.add_argument("--sims", "-s", help="path to sim pickle files, give in same order as noise sources, pass sum last ", nargs = "+")
    args = parser.parse_args()

    config_path = find_config(args.sims[0])
    config = read_config(config_path)
    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    channel_ids = config["channels_to_include"]
    electronic_temperature = config["electronic_temperature"]


    bandpass_filt = channelBandPassFilter()
    

    spectrum_fitter = spectrumFitter(args.data, args.sims[:-1])
    fit_results = spectrum_fitter.get_fit_gain(mode="electronic_temp")


    plt.style.use("gaudi")

    pdf = PdfPages("test_avg_ft.pdf") 
    for idx, channel_id in enumerate(channel_ids):
        fit_param = fit_results[idx]
        print(fit_param)
        gain_factor = fit_param[0].value
        temp_factor = fit_param[1].value
        gain_error = fit_param[0].error
        
        fit_function = spectrum_fitter.get_fit_function(mode="electronic_temp", channel_id=channel_id)

        fig, ax = plt.subplots()
        labels = noise_sources
        # if include_sum:
        #     labels += ["sum"]
        
        if args.data is not None:
            frequencies, frequency_spectrum = read_freq_spec_file(args.data)
            butter_filt = bandpass_filt.get_filter(frequencies, station_id=23, channel_id=channel_id, det=0, passband=[0.1, 0.8], filter_type="butter", order=2)
            frequency_spectrum = frequency_spectrum * np.abs(butter_filt)
            ax.plot(frequencies, frequency_spectrum[channel_id], label = "data")
        
        for i, sim in enumerate(args.sims[:-1]):
            if noise_sources[i] == "electronic":
                labels[i] += f" (T = {electronic_temperature} K)"
            elif noise_sources[i] == "ice":
                labels[i] += " (T = ~240 K)"
            elif labels[i] == "sum":
                labels[i] += f" (scaled by {convert_to_db(gain_factor):.0f} dB)"
            frequencies, frequency_spectrum = read_freq_spec_file(sim)
            
            frequency_spectrum = frequency_spectrum * gain_factor
                
            ax.plot(frequencies, frequency_spectrum[channel_id], label=labels[i])

        ax.plot(frequencies, smooth(fit_function(frequencies, gain_factor, temp_factor)), label="fit_result")
        
        ax.legend()
        ax.set_title(f"Channel {channel_id}")
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("spectral amplitude / V/GHz")
        ax.set_xlim(0, 1.)
        fig.savefig(pdf, format="pdf")

    pdf.close()
