import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter

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
    parser.add_argument("--station", "-s", type=int, default=24)
    parser.add_argument("--channel", "-c", type=int, default=0)
    parser.add_argument("--run", "-r", type=int, default=100)
    parser.add_argument("--event", "-e", type=int, default=6)
    args = parser.parse_args()

    reader = readRNOGData()
    run_dir = glob.glob(f"{os.environ['RNO_G_DATA']}/station{args.station}/run{args.run}")
    selectors = [lambda event_info : event_info.triggerType == "FORCE"]
    reader.begin(run_dir,
                 read_calibrated_data=False,
                 selectors=selectors,
                 max_trigger_rate=2*units.Hz,
                 convert_to_voltage=True)

    
    for event in reader.run():
        station = event.get_station()
        channel = station.get_channel(args.channel)
        # freqs are given in GHz (NuRadio's base frequency unit)
        freq = channel.get_frequencies()
        spec = channel.get_frequency_spectrum()
        print(spec.shape)
        break
    
        
    passband = [0.1, 0.7]
    bandpass = channelBandPassFilter()
    start_order = 10
    filt_start = bandpass.get_filter(freq, args.station, args.channel, 0, passband = [passband[0], 1.6], filter_type="butter", order=start_order)
    end_order = 10
    filt_end = bandpass.get_filter(freq, args.station, args.channel, 0, passband = [0., passband[-1]], filter_type="butter", order=end_order)

    selection = np.where(np.logical_and(0.2 < freq, freq < 0.5))
    spec = smooth(np.abs(spec))
    spec_normalized = spec / np.mean(spec[selection]) 


    spec_filt = spec * filt_start
    spec_filt *= filt_end
    spec_filt = np.abs(spec_filt)
    spec_filt = spec_filt / np.mean(spec_filt[selection])


    fig, ax = plt.subplots()
    ax.plot(freq, spec_normalized, label = "unfiltered")
    ax.plot(freq, np.abs(filt_start), label = f"butter, o={start_order}")
    ax.plot(freq, np.abs(filt_end), label = f"butter, o={end_order}")
    ax.plot(freq, spec_filt, label = f"filtered spectrum")
    ax.set_title(f"Application of butterworth filter {passband} GHz")
    ax.hlines(0, 0, 2, ls="dashed", color="black")
    ax.legend()
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("spectral amplitude / V/GHz")
    fig.savefig(f"single_spec_s{args.station}_c{args.channel}_run{args.run}_e{args.event}")
