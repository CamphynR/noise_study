import argparse
import datetime
import json
import logging
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.detector import antennapattern
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import units

def get_sampled_VEL_from_pickle(pickle_path, channel=0):
    antenna_pattern = antennapattern.get_pickle_antenna_response(pickle_path)
#    antenna_pattern = antenna_provider.load_antenna_pattern("createLPDA_100MHz_z1cm_InFirn_RG")
    (           ori_theta,
                ori_phi,
                rot_theta,
                rot_phi,
                frequencies,
                thetas,
                phis,
                RVEL_phi,
                RVEL_theta,
            ) = antenna_pattern


    diff_phis = np.diff(phis)
    diff_phis = diff_phis[diff_phis!=0]
    diff_phis = diff_phis[diff_phis!=-2*np.pi]

    if channel in channel_mapping["VPol"]:
        zenith_max = 90 * units.degree
        pol = "theta"
    elif channel in channel_mapping["HPol"]:
        zenith_max = 90 * units.degree
        pol = "phi"
    else:
        zenith_max = orientation[0]
        pol = "theta"
    
    dtheta = np.diff(thetas)[0]
    dphi = np.diff(phis)
    dphi = dphi[dphi!=0]
    dphi = dphi[0]

    zeniths = np.linspace(0 * units.degree, 180 * units.degree, 100)
    azimuths = np.linspace(0 * units.degree , 360 * units.degree, 64)
    VELs = []
    zeniths_proper = []
    for zenith in zeniths:
        vel = []
        for azimuth in azimuths:
#            indices = np.where((thetas == zenith) & (phis == azimuth))
            indices = np.where((np.isclose(thetas,zenith,atol=dtheta/2.)) & (np.isclose(phis,azimuth,atol=dphi/2.)))

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
    

    return zenith_idx, freqs, VELs

def get_sampled_VEL(antenna_model, channel=0, interpolation_method="complex",
                    zeniths=np.linspace(0 * units.degree, 180 * units.degree, 37),
                    azimuths=np.linspace(0 * units.degree , 360 * units.degree, 10)
                    ):
    antenna_provider = AntennaPatternProvider()
    antenna_pattern = antenna_provider.load_antenna_pattern(antenna_model, interpolation_method=interpolation_method)
#    antenna_pattern = antenna_provider.load_antenna_pattern("createLPDA_100MHz_z1cm_InFirn_RG")


    if channel in channel_mapping["VPol"]:
        zenith_max = 90 * units.degree
#        zenith_max = 1.57079633
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 1000)
        pol = "theta"
    elif channel in channel_mapping["HPol"]:
        zenith_max = 90 * units.degree
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 100)
        pol = "phi"
    else:
        zenith_max = orientation[0]
        freqs = np.linspace(0 * units.MHz, 1000 * units.MHz, 1000)
#        freqs = np.fft.rfftfreq(2048, d=1./3.2)
        pol = "theta"

    zenith_idx = np.where(np.isclose(zeniths, zenith_max, atol=np.diff(zeniths)[0]/2.))[0][0]


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
    parser.add_argument("--use_db", action="store_true")
    parser.add_argument("--save_boresight_max", action="store_true", help="for plotting in antenna systematics")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    detector = Detector(select_stations=args.station)
    detector_time = datetime.datetime(2023,12,31)
    detector.update(detector_time)

    channel_mapping = {"VPol" : [0, 1, 2, 3, 6, 7, 9, 10, 22, 23],
                       "HPol" : [4, 8, 11, 21],
                       "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    orientation = detector.get_antenna_orientation(args.station, args.channel)
    pol = "theta" if args.channel in channel_mapping["VPol"] else "phi"


    # 37 to make sure we have zenith=90
    zeniths = np.linspace(0 * units.degree, 180 * units.degree, 37)
    azimuths = np.linspace(0 * units.degree , 360 * units.degree, 10)


    if args.use_db:
        # VEL loaded from NuRadio's AntennaPatternProvider
        freqs_list = []
        zenith_max_idx_list = []
        VEL_models = []
        for antenna_model in args.antenna_models:
            if antenna_model.startswith("RNOG_vpol_v4"):
                interpolation_method = "complex"
            else:
                interpolation_method = "complex"
            zenith_max_idx, freqs, VEL = get_sampled_VEL(antenna_model, args.channel,
                                                       interpolation_method=interpolation_method,
                                                       zeniths=zeniths,
                                                       azimuths=azimuths)
            zenith_max_idx_list.append(zenith_max_idx)
            print(antenna_model)
            print(zeniths[zenith_max_idx]/units.degree)
            freqs_list.append(freqs)
            VEL_models.append(VEL)
    
    else:
        # VEL directly read from pickle
        antenna_dir = "/user/rcamphyn/software/NuRadioMC/NuRadioReco/detector/AntennaModels/"
        pickle_paths = [antenna_dir + ant_name + "/" + ant_name + ".pkl" for ant_name in args.antenna_models]
        freqs_list = []
        zenith_max_idx_list = []
        VEL_models = []
        for i, pickle in enumerate(pickle_paths):
            logger.info(f"processing {args.antenna_models[i]}")
            zenith_max_idx, freqs, VEL = get_sampled_VEL_from_pickle(pickle, args.channel)
            zenith_max_idx_list.append(zenith_max_idx)
            freqs_list.append(freqs)
            VEL_models.append(VEL)


    plt.style.use("retro")

    if args.save_boresight_max:
        save_path = "sim/library/antenna_models_max.json"
        antenna_models_max = {}

    # PLOT AT MAX BORESIGHT
    # ---------------------
    fig, ax = plt.subplots(figsize=(20, 10))
    pol_idx = 0 if pol=="theta" else 1
    for i, VELs in enumerate(VEL_models):
        # NuRadio default
        if args.antenna_models[i] == "RNOG_vpol_v3_5inch_center_n1.74":
            ls = "dashed"
            lw = 1.5
        else:
            ls = None
            lw = 1.
        ax.plot(freqs_list[i]/units.MHz, VELs[zenith_max_idx_list[i], 0, pol_idx], label=args.antenna_models[i],
                lw=lw, ls=ls)
        if args.save_boresight_max:
            antenna_models_max[args.antenna_models[i]] = np.max(VELs[zenith_max_idx_list[i], 0, pol_idx])
            print(antenna_models_max)


    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("VEL / m")
    ax.legend()
    ax.set_title("VEL comparisons")
    ax.set_xlim(0., 1200.)
    fig.savefig(f"figures/tests/antenna_vel_comparison", bbox_inches="tight")


    if args.save_boresight_max:
        with open(save_path, "w") as file:
            json.dump(antenna_models_max, file)



    # PLOT OVER DIFFERENT ZENITHS
    # ---------------------------
    azimuth_fixed = 0 * units.degree
    azimuth_idx = np.where(np.isclose(azimuths, azimuth_fixed, atol=np.diff(azimuths)[0]/2.))[0][0]

    fig, ax = plt.subplots()
    VEL_models = np.array(VEL_models)
    for i, VEls in enumerate(VEL_models):
        base_zenith_line = ax.plot(freqs_list[i]/units.MHz,
                                   VEL_models[i, -1, azimuth_idx, pol_idx],
                                   label=args.antenna_models[i])

        color = base_zenith_line[0].get_color()

        for zenith_i, zenith in enumerate(zeniths[0:-1]):
            if not zenith_i % 10:
                ax.plot(freqs_list[i]/units.MHz,
                        VEL_models[i, zenith_i, azimuth_idx, pol_idx],
                        color=color,
                        alpha=zenith_i/len(zeniths),
    #                    label=f"args.antenna_models[i]\nzenith={zeniths[zenith_i]}",
                        )

    norm = mcolors.Normalize(vmin=zeniths[0], vmax=zeniths[-1])
    cmap = cm.Grays
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    rad_to_deg = lambda rad, pos : "{:.2f}".format(rad * (180/np.pi))
    fmt = FuncFormatter(rad_to_deg)
    plt.colorbar(sm,label="zenith / deg", ax=ax, format=fmt)

    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("VEL / m")
    ax.legend()
    ax.set_title("VEL comparisons")
    ax.set_xlim(0., 1200.)
    fig.savefig(f"figures/tests/antenna_vel_comparison_zeniths", bbox_inches="tight")
        


