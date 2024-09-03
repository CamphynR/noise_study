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
                        default = None)
    parser.add_argument("-y", "--year",
                        type = int,
                        default = 2023)
    args = parser.parse_args()

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(args.year, 8, 1))

    if args.channel is not None:
        response = det.get_signal_chain_response(station_id = args.station, channel_id = args.channel)
        fig, ax = response.plot(det.get_signal_chain_response(station_id = args.station, channel_id = args.channel))
    else:
        responses = [det.get_signal_chain_response(station_id = args.station, channel_id = c) for c in range(7)]
        freq = np.linspace(0, 1.4) * units.GHz
        gains = [response(freq, component_names = "total") for response in responses]
        fig, ax = plt.subplots()
        for i, gain in enumerate(gains):
            mask = gain > 0
            ax.plot(freq[mask]/units.MHz, 20 * np.log10(np.abs(gain[mask])), label = f"total of channel {i}")
        ax.set_xlabel("freq / MHz")
        ax.set_ylabel("gain / dB")
        ax.legend(loc = 1)

    fig.suptitle(f"response station {args.station}, year {args.year}")
    channel_name = f"channel{args.channel}" if args.channel is not None else "allchannels"
    fig_path = os.path.abspath(f"{os.path.dirname(__file__)}/../figures")
    fig.savefig(f"{fig_path}/response_station{args.station}_{channel_name}_year{args.year}.png") 