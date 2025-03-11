import argparse
import matplotlib.pyplot as plt
import numpy as np


from utilities.utility_functions import read_pickle



def read_average_freq_spectrum_from_pickle(file : str):
    contents = read_pickle(file)
    return contents["freq"], contents["frequency_spectrum"], contents["var_frequency_spectrum"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    args = parser.parse_args()

    frequency, frequency_spectrum, var_frequency_spectrum = read_average_freq_spectrum_from_pickle(args.pickle)
    
    channel_id = 0
    plt.plot(frequency, frequency_spectrum[channel_id])
    plt.show()