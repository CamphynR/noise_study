import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
from utility_functions import rms_from_temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", "-t", type = int, default = 300)
    parser.add_argument("--with_detector", action="store_true")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    
    stations = [12, 13, 21, 22, 23, 24]
    detector = detector.Detector(source="rnog_mongo",
                                 always_query_entire_description=False,
                                 database_connection="RNOG_public",
                                 select_stations=stations,
                                 log_level=logging.INFO)
    
    detector.update(Time(config["detector_time"]))

    Vrms = [[rms_from_temp(args.temp, detector, station, channel=ch, include_amps=args.with_detector) for station in stations]
            for ch in range(24)]

    fig, ax = plt.subplots()
    ax.plot(np.array(Vrms) / units.mV, label = [str(s) for s in stations])
    ax.legend(loc = "upper left", bbox_to_anchor=(1.1, 1.1))
    ax.set_xticks(list(range(24)), rotation = 45)
    ax.set_xlabel("channel")
    ax.set_ylabel("V / mV")
    ax.set_ylim(0.005, 0.01)
    fig.tight_layout()
    fig.savefig("compare_stations_vrms_from_temp")
