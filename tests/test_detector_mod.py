import argparse
import datetime
import logging
import numpy as np

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.utilities import units


def get_antenna_type(ch_id):
    if ch_id in [0, 1, 2, 3]:
        return "VPol"
    elif ch_id in [4, 8]:
        return "HPol"
    else:
        return "LPDA"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=23, type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    station_id = args.station
    include_channels = [0, 1, 2, 3, 4, 8]


    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74"}

# all kwargs explicitly defined
#    detector = ModDetector(database_connection='RNOG_public', log_level=logging.NOTSET, over_write_handset_values=None,
#                 database_time=None, always_query_entire_description=False, detector_file=None,
#                 select_stations=station_id, create_new=False)

# only subset of kwargs defined
    detector = ModDetector(database_connection='RNOG_public', select_stations=station_id)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)
    print(detector.get_site_coordinates())

    for channel_id in [0, 1, 2, 3]:
        antenna_model = "RNOG_vpol_v3_5inch_center_n1.74"
        detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)
        print("model getter yields: " + detector.get_antenna_model(station_id, channel_id))
        ch_info = detector.get_channel(args.station, channel_id)
        print("channel info yields: " + ch_info["signal_chain"]["VEL"])


    for channel_id in [4, 8]:
        antenna_model = "RNOG_hpol_v4_8inch_center_n1.74"
        detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)
