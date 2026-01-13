import argparse
import matplotlib.pyplot as plt
import numpy as np
from utilities.utility_functions import read_pickle


from NuRadioReco.utilities import units





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to pickle file")
    args = parser.parse_args()

    channel_id = 0


    pickle_contents = read_pickle(args.path)
    bin_centres = pickle_contents["header"]["bin_centres"]
    bin_width = np.diff(bin_centres)[0]
    vrms_histograms = pickle_contents["vrms_histograms"]



    plt.style.use("retro")
    fig, ax = plt.subplots()
    ax.bar(bin_centres/units.mV, vrms_histograms[channel_id], width=bin_width/units.mV)
    ax.set_xlabel("Vrms / mV")
    plt.savefig("test")
    
