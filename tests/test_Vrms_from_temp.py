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
    parser.add_argument("--station", "-s", type = int, default = 24)
    parser.add_argument("--with_detector", action="store_true")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    test_temperatures = np.arange(200, 500, 50)
    detector = detector.Detector(source="rnog_mongo",
                                 always_query_entire_description=False,
                                 database_connection="RNOG_public",
                                 select_stations=args.station,
                                 log_level=logging.INFO)
    
    detector.update(Time(config["detector_time"]))

    Vrms = [rms_from_temp(test_temperatures, detector, args.station, channel=ch,include_amps=args.with_detector)
            for ch in range(24)]

    fig, ax = plt.subplots()
    ax.plot(np.array(Vrms) / units.mV, label = [str(T) for T in test_temperatures])
    ax.legend(loc = "upper left", bbox_to_anchor=(1.1, 1.1))
    ax.set_title(f"station {args.station}")
    ax.set_xlabel("channel")
    ax.set_ylabel("V / mV")
    fig.tight_layout()
    fig.savefig("test_vrms_from_temp")
