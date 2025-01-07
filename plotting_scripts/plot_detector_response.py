import datetime
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="%(prog)s",
                                     usage = "script to plot detector response from hardware database")
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 24)
    parser.add_argument("-c", "--channel",
                        type = int,
                        nargs = "+",
                        default = [])
    parser.add_argument("-y", "--year",
                        type = int,
                        default = 2023)
    parser.add_argument("--min_gain", type = float, default = 0.1)
    args = parser.parse_args()

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(args.year, 8, 1))

    if len(args.channel) == 1:
        response = det.get_signal_chain_response(station_id = args.station, channel_id = args.channel[0])
        fig, ax = response.plot()
    else:
        args.channel = np.arange(24)
        responses = [det.get_signal_chain_response(station_id = args.station, channel_id = c) for c in args.channel]
        freq = np.linspace(0, 1.4) * units.GHz
        gains = [response(freq, component_names = "total") for response in responses]
        max_gain = np.max(np.abs(gains))
        fig, ax = plt.subplots(figsize = (14, 8))
        for i, gain in enumerate(gains):
            mask = np.abs(gain) > 0
            ax.plot(freq[mask]/units.MHz, 20 * np.log10(np.abs(gain[mask])), label = f"total of channel {args.channel[i]}")
        ax.hlines(args.min_gain * 20 * np.log10(max_gain), 0., 1000., ls = "dashed", color = "red", label = f"{args.min_gain} * max_gain")
        ax.set_xlabel("freq / MHz")
        ax.set_ylabel("gain / dB")
        ax.legend(loc = "upper left", bbox_to_anchor = (1.02, 1.), ncol = 2, fontsize = "small")

    fig.suptitle(f"response station {args.station}, year {args.year}")
    fig.tight_layout()
    fig_path = os.path.abspath(f"{os.path.dirname(__file__)}/../figures")
    if len(args.channel) == 24:
        channels_string = "all_channels"
    else:
        channels_string = "ch_" + "_".join(str(c) for c in args.channel)
    figname = f"{fig_path}/response_station{args.station}_{channels_string}_year{args.year}.png"
    print(f"Saving as {figname}")
    fig.savefig(figname) 
