import argparse
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import os

from flower.utils import read_flower_data, flowerDataset
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import sinewave_subtraction



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folders", nargs="+")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()


    plt.style.use("retro")
    station_id = 22
    nr_samples = 1024
    sampling_rate = 0.472

    freqs = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    passband = [0.01, 0.95*np.max(freqs)]
    print(0.95*np.max(freqs))
    bandpass_filter = channelBandPassFilter()
    bandpass_filter = bandpass_filter.get_filter(freqs, station_id=-1, channel_id=-1, det=0,
                                                 passband=passband,
                                                 filter_type="gaussian_tapered")
    plt.plot(freqs, np.abs(bandpass_filter))
    plt.savefig("test")


    radiant_v3_install_date = Time("2024-07-14", format="isot")
    print(radiant_v3_install_date.unix)

#    save_folder = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/flower_average_ft/station{station_id}/"
    save_folder = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/flower_average_ft/station{station_id}/"
    if args.test:
        save_folder = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/flower_average_ft/station_test/"

    for run_path in args.run_folders:
        if os.path.isdir(run_path):
            basename = os.path.basename(run_path)
        else:
            basename = os.path.basename(run_path).split(".")[0]
        savename = "average_ft_" + basename + ".pickle"
        ds = flowerDataset(run_path, read_data_in_volts=True, trigger="PPS") 
#        print(ds.keys())
#        print(ds["flower_gains"])
#        print(np.array(ds["events"][0]["ch0"]))
            
        if ds.run_start_time > radiant_v3_install_date.unix:
            print("saving")
            sinewavesubtraction_kwargs = dict(
                    peak_prominence = 3.,
                    sampling_rate = 0.472,
                    freq_band = passband
                    )
            def sinewave_subtraction_only_wf(wf, **sinewavesubtraction_kwargs):
                return sinewave_subtraction(wf, **sinewavesubtraction_kwargs)[0]
            ds.apply_function_to_wfs(sinewave_subtraction_only_wf, **sinewavesubtraction_kwargs)
            ds.apply_function_to_wfs(sinewave_subtraction_only_wf, **sinewavesubtraction_kwargs)

            ds.save_average_spectrum(save_folder + savename, filt=bandpass_filter, debug=args.test)

        if args.test:
            wf_idx = 0
            channel_id = 0
            fig, ax = plt.subplots()
            ds.plot_wf(wf_idx, channel_id, ax)
            fig.savefig("figures/tests/test_flower_wf.png")
            plt.close()
