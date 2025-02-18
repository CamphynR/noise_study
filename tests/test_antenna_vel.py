import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.utilities import units

if __name__ == "__main__":
    
    station_id = 23
    channel_id = 0
    zenith = 90 * units.degree
    azimuth = 0 * units.degree
    
    sampling_rate = 3.2*units.GHz

    freqs = np.fft.rfftfreq(2048, d=1./sampling_rate)
    
    
    def get_antenna_type(ch_id):
        if ch_id in [0, 1, 2, 3]:
            return "VPol"
        elif ch_id in [4, 8]:
            return "HPol"
        else:
            return "LPDA"


        
    

    detector = ModDetector(database_connection='RNOG_public',
                           log_level=logging.NOTSET, over_write_handset_values=None,
                           database_time=None, always_query_entire_description=False, detector_file=None,
                           select_stations=station_id, create_new=False)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    antenna_model = detector.get_antenna_model(station_id, channel_id)
    orientation = detector.get_antenna_orientation(station_id, channel_id)

    antenna_provider = AntennaPatternProvider()
    antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model)
    pol = "phi" if channel_id in [4, 8] else "theta"
    ant_type = "HPol" if channel_id in [4, 8] else "VPol"
    VEL = np.abs(antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)[pol])
    
    antenna_models_new = {"HPol" : "RNOG_hpol_v4_8inch_center_n1.74",
                          "VPol" : "RNOG_vpol_v3_5inch_center_n1.74"}

    detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_models_new[ant_type])
    antenna_model_new = detector.get_antenna_model(station_id, channel_id)


    antenna_pattern_new = antenna_provider.load_antenna_pattern(antenna_model_new)
    VEL_new = np.abs(antenna_pattern_new.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)[pol])

    plt.plot(freqs, VEL, label = antenna_model)
    plt.plot(freqs, VEL_new, label = antenna_model_new)
    plt.legend()
    plt.xlabel("freqs / GHz")
    plt.ylabel("VEL")
    plt.title(f"{ant_type} zenith {zenith / units.degree} deg, azimuth {azimuth / units.degree} deg")
    plt.savefig(f"figures/test_antenna_vel_{ant_type}") 
