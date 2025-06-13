import argparse
import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import numpy as np
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import units

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=11)
    parser.add_argument("--channel", "-c", type=int, default=0)
    args = parser.parse_args()
    


    channel_mapping = {"VPol" : [0, 1, 2, 3, 6, 7, 9, 10, 22, 23],
                       "HPol" : [4, 8, 11, 21],
                       "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}


        
    # for reference, these should be the latest antenna files (written 2025/05/22)
    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74",
                      "LPDA" : "createLPDA_100MHz_InfFirn_n1.4"}
    
    detector = Detector(select_stations=args.station)
    detector_time = datetime.datetime(2023,12,31)
    detector.update(detector_time)


    antenna_model = detector.get_antenna_model(args.station, args.channel)
    print(antenna_model)
    orientation = detector.get_antenna_orientation(args.station, args.channel)

    antenna_provider = AntennaPatternProvider()
    antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model)

    zeniths = np.linspace(0 * units.degree, 180 * units.degree, 7)

    if args.channel in channel_mapping["VPol"]:
        zenith_max = 90 * units.degree
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 100)
        pol = "theta"
    elif args.channel in channel_mapping["HPol"]:
        zenith_max = 90 * units.degree
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 100)
        pol = "phi"
    else:
        zenith_max = orientation[0]
        freqs = np.linspace(0 * units.MHz, 200 * units.MHz, 100)
        pol = "theta"
    zenith_idx = np.where(np.isclose(zeniths, zenith_max, atol=180*units.degree/7/2))[0][0]

    azimuths = np.linspace(0 * units.degree , 360 * units.degree, 180)
    VELs = []
    for zenith in zeniths:
        vel = []
        for azimuth in azimuths:
            VEL = antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)
            vel.append([np.abs(VEL["theta"]), np.abs(VEL["phi"])])
        VELs.append(vel)
    
    VELs = np.array(VELs)



    plt.style.use("retro")
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), subplot_kw={"projection" : "polar"})
    pol_theta = axs[0].pcolormesh(azimuths, freqs/units.MHz, np.moveaxis(VELs[zenith_idx, :, 0], 0, 1), vmax=np.max(VELs))
    axs[0].set_title("VEL theta")
    pol_phi = axs[1].pcolormesh(azimuths, freqs/units.MHz, np.moveaxis(VELs[zenith_idx, :, 1], 0, 1), vmax=np.max(VELs))
    axs[1].set_title("VEL phi")
    fig.colorbar(pol_theta, ax=axs, location="bottom", shrink=0.6, label="|VEL|")
    fig.savefig(f"figures/tests/antenna_vel_ch_{args.channel}", bbox_inches="tight")


    fig, ax = plt.subplots(figsize=(20, 10))
    pol_idx = 0 if pol=="theta" else 1
    ax.plot(freqs/units.MHz, VELs[zenith_idx, 0, pol_idx])
    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("VEL / a.u.")
    ax.set_title("VEL sliced")
    fig.savefig(f"figures/tests/antenna_vel_ch_{args.channel}_sliced", bbox_inches="tight")


    
#    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
#    for i, zenith in enumerate(zeniths):
#        for j, f in enumerate(freqs):
#            axs[i].plot(azimuths/units.degree, VELs[:, i, j], label = f"freq = {f} GHz")
#        axs[i].set_title(f"zenith = {zenith/units.degree:.0f} degrees")
#        axs[i].set_xlabel("azimuth / degree")
#        axs[i].set_ylabel("VEL")
#    axs[i].legend(bbox_to_anchor=(1.1, 1.1))
#    fig.suptitle(ant_type)
#    fig.tight_layout()
#    plt.savefig(f"tests/test_antenna_vel_{ant_type}") 
#
##    fig, ax = plt.subplots(subplot_kw={'projection' : 'polar'})
##    pol = ax.pcolormesh(azimuths, freqs/units.MHz, VELs.T)
##    plt.title(f"{ant_type} zenith {zenith / units.degree} deg")
#
##    for azimuth in azimuths:
##        zenith = 30 * units.degree
##
##        VEL = np.abs(antenna_pattern_new.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)[pol])
##        VELs.append(vel)
##    
##    VELs = np.array(VELs)
##    print(VELs.shape)
##
##    
##    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
##    for i, zenith in enumerate(zeniths):
##        for j, f in enumerate(freqs[::int(len(freqs)/5)]):
##            axs[i].plot(azimuths/units.degree, VELs[:, i, j], label = f"freq = {f} GHz")
##        axs[i].set_title(f"zenith = {zenith/units.degree:.0f} degrees")
##        axs[i].set_xlabel("azimuth / degree")
##        axs[i].set_ylabel("VEL")
##    axs[i].legend(bbox_to_anchor=(1.1, 1.1))
##    fig.tight_layout()
##
