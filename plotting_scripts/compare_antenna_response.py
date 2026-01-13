import argparse
import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import numpy as np
from NuRadioReco.detector import antennapattern
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import units

def get_sampled_VEL_from_pickle(pickle_path, channel=0):
    antenna_pattern = antennapattern.get_pickle_antenna_response(pickle_path)
#    antenna_pattern = antenna_provider.load_antenna_pattern("createLPDA_100MHz_z1cm_InFirn_RG")
    (           zen_boresight,
                azi_boresight,
                zen_ori,
                azi_ori,
                frequencies,
                thetas,
                phis,
                RVEL_phi,
                RVEL_theta,
            ) = antenna_pattern
    print(frequencies[0:100])

    zeniths = np.linspace(0 * units.degree, 180 * units.degree, 45)

    if channel in channel_mapping["VPol"]:
        zenith_max = 90 * units.degree
        pol = "theta"
    elif channel in channel_mapping["HPol"]:
        zenith_max = 60 * units.degree
        pol = "phi"
    else:
        zenith_max = orientation[0]
        pol = "theta"

    print(thetas[:100])
    azimuths = np.linspace(0 * units.degree , 360 * units.degree, 90)
    VELs = []
    zeniths_proper = []
    for zenith in zeniths:
        vel = []
        for azimuth in azimuths:
            indices = np.where((thetas == zenith) & (phis == azimuth))
            if len(frequencies[indices]) == 0:
                continue
            VEL_theta = RVEL_theta[indices]
            VEL_phi = RVEL_phi[indices]
            vel.append([np.abs(VEL_theta), np.abs(VEL_phi)])
        vel = np.array(vel)
        if len(vel) !=0:
            VELs.append(vel) 
            zeniths_proper.append(zenith)

    VELs = np.array(VELs)
    freqs = frequencies[indices]

    zenith_idx = np.where(np.isclose(zeniths_proper, zenith_max, atol=180*units.degree/7/2))[0][0]
    print(zenith_idx)
    

    return zenith_idx, freqs, VELs

def get_sampled_VEL(antenna_model, channel=0):
    antenna_provider = AntennaPatternProvider()
    antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model, interpolation_method="magphase")
#    antenna_pattern = antenna_provider.load_antenna_pattern("createLPDA_100MHz_z1cm_InFirn_RG")

    zeniths = np.linspace(0 * units.degree, 180 * units.degree, 45)

    if channel in channel_mapping["VPol"]:
#        zenith_max = 90 * units.degree
        zenith_max = 1.57079633
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 500)
        pol = "theta"
    elif channel in channel_mapping["HPol"]:
        zenith_max = 60 * units.degree
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 100)
        pol = "phi"
    else:
        zenith_max = orientation[0]
        freqs = np.linspace(0 * units.MHz, 250 * units.MHz, 1000)
#        freqs = np.fft.rfftfreq(2048, d=1./3.2)
        pol = "theta"
    zenith_idx = np.where(np.isclose(zeniths, zenith_max, atol=180*units.degree/7/2))[0][0]

    azimuths = np.linspace(0 * units.degree , 360 * units.degree, 10)
    VELs = []
    for zenith in zeniths:
        vel = []
        for azimuth in azimuths:
            VEL = antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)
            vel.append([np.abs(VEL["theta"]), np.abs(VEL["phi"])])
        VELs.append(vel) 
    VELs = np.array(VELs)

    return zenith_idx, freqs, VELs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--antenna_models", nargs = "+", type=str, default=None)
    parser.add_argument("--station", type=int, default=11)
    parser.add_argument("--channel", type=int, default=0, help="used to determine oreintation of VEL maximum")
    args = parser.parse_args()
    

    detector = Detector(select_stations=args.station)
    detector_time = datetime.datetime(2023,12,31)
    detector.update(detector_time)

    channel_mapping = {"VPol" : [0, 1, 2, 3, 6, 7, 9, 10, 22, 23],
                       "HPol" : [4, 8, 11, 21],
                       "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    orientation = detector.get_antenna_orientation(args.station, args.channel)
    pol = "theta" if args.channel in channel_mapping["VPol"] else "phi"

    # VEL loaded from NuRadio's AntennaPatternProvider
#    freqs_list = []
#    zenith_idx_list = []
#    VEL_models = []
#    for antenna_model in args.antenna_models:
#        zenith_idx, freqs, VEL = get_sampled_VEL(antenna_model, args.channel)
#        zenith_idx_list.append(zenith_idx)
#        freqs_list.append(freqs)
#        VEL_models.append(VEL)
    
    # VEL directly read from pickle
    antenna_dir = "/user/rcamphyn/software/NuRadioMC/NuRadioReco/detector/AntennaModels/"
    pickle_paths = [antenna_dir + ant_name + "/" + ant_name + ".pkl" for ant_name in args.antenna_models]
    freqs_list = []
    zenith_idx_list = []
    VEL_models = []
    for pickle in pickle_paths:
        zenith_idx, freqs, VEL = get_sampled_VEL_from_pickle(pickle, args.channel)
        zenith_idx_list.append(zenith_idx)
        freqs_list.append(freqs)
        VEL_models.append(VEL)


    plt.style.use("retro")

    fig, ax = plt.subplots(figsize=(20, 10))
    pol_idx = 0 if pol=="theta" else 1
    for i, VELs in enumerate(VEL_models):
        ax.plot(freqs_list[i]/units.MHz, VELs[zenith_idx_list[i], 0, pol_idx], label=args.antenna_models[i])
    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("VEL / a.u.")
    ax.legend()
    ax.set_xlim(0,1000.)
    ax.set_title("VEL comparisons")
    fig.savefig(f"figures/tests/antenna_vel_comparison", bbox_inches="tight")
