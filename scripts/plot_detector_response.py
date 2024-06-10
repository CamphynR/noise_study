import datetime
import argparse
import os.path
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
                        default = 0)
    parser.add_argument("-y", "--year",
                        type = int,
                        default = 2023)
    args = parser.parse_args()

    det = detector.Detector(source = "rnog_mongo",
                            always_query_entire_description = False,
                            database_connection = "RNOG_public",
                            select_stations = args.station)
    det.update(datetime.datetime(args.year, 8, 1))
    response = det.get_signal_chain_response(station_id = args.station, channel_id = args.channel)
    fig, ax = response.plot()
    fig_path = os.path.abspath(f"{os.path.dirname(__file__)}/../figures")
    fig.savefig(f"{fig_path}/test_station{args.station}_year{args.year}.png")