import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.utilities import units

if __name__ == "__main__":
    
    station_id = 23
    channel_id = 4
    zenith = 120 * units.degree
    azimuth = 0 * units.degree
    azimuths = np.linspace(0*units.degree , 360*units.degree, 180)
    
    sampling_rate = 3.2*units.GHz

    zeniths = np.array([30, 90, 120]) * units.degree
    
    
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
    
    antenna_models_new = {"HPol" : "RNOG_hpol_v4_8inch_center_n1.74",
                          "VPol" : "RNOG_vpol_v3_5inch_center_n1.74"}

    detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_models_new[ant_type])
    antenna_model_new = detector.get_antenna_model(station_id, channel_id)


    antenna_pattern_new = antenna_provider.load_antenna_pattern(antenna_model_new)
    if ant_type =="VPol":
        freqs = np.linspace(0.1, 0.5, 5)
    else:
        freqs = np.linspace(0.1, 0.8, 8)


    print(freqs)

    VELs = []
    for azimuth in azimuths:
        vel = []
        for zenith in zeniths:
            VEL = np.abs(antenna_pattern_new.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)[pol])
            vel.append(VEL)
        VELs.append(vel)
    
    VELs = np.array(VELs)
    print(VELs.shape)

    
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    for i, zenith in enumerate(zeniths):
        for j, f in enumerate(freqs):
            axs[i].plot(azimuths/units.degree, VELs[:, i, j], label = f"freq = {f} GHz")
        axs[i].set_title(f"zenith = {zenith/units.degree:.0f} degrees")
        axs[i].set_xlabel("azimuth / degree")
        axs[i].set_ylabel("VEL")
    axs[i].legend(bbox_to_anchor=(1.1, 1.1))
    fig.suptitle(ant_type)
    fig.tight_layout()
    plt.savefig(f"tests/test_antenna_vel_{ant_type}") 

#    fig, ax = plt.subplots(subplot_kw={'projection' : 'polar'})
#    pol = ax.pcolormesh(azimuths, freqs/units.MHz, VELs.T)
#    plt.title(f"{ant_type} zenith {zenith / units.degree} deg")

#    for azimuth in azimuths:
#        zenith = 30 * units.degree
#
#        VEL = np.abs(antenna_pattern_new.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)[pol])
#        VELs.append(vel)
#    
#    VELs = np.array(VELs)
#    print(VELs.shape)
#
#    
#    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
#    for i, zenith in enumerate(zeniths):
#        for j, f in enumerate(freqs[::int(len(freqs)/5)]):
#            axs[i].plot(azimuths/units.degree, VELs[:, i, j], label = f"freq = {f} GHz")
#        axs[i].set_title(f"zenith = {zenith/units.degree:.0f} degrees")
#        axs[i].set_xlabel("azimuth / degree")
#        axs[i].set_ylabel("VEL")
#    axs[i].legend(bbox_to_anchor=(1.1, 1.1))
#    fig.tight_layout()
#
