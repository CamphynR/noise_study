import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import mattak.Dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-d", "--data_path",
                        default = None)
    parser.add_argument('-s', "--station",
                        type = int)
    parser.add_argument("-r", "--run",
                        type = int)
    parser.add_argument('-e', "--event",
                        type = int, default = 0)
    parser.add_argument("-c", "--channel",
                        type = int, default = 0)
    args = parser.parse_args()

    ds = mattak.Dataset.Dataset(station = args.station, run = args.run, data_path = args.data_path,  backend = "pyroot", verbose = True)
    ds.setEntries(args.event)
    print(f"dataset contains {ds.N()} entries")

    wfs = ds.wfs()
    print(f"shape of wfs is {wfs.shape}")

    plt.plot(wfs[args.channel])
    plt.xlabel("sample")
    plt.ylabel("ADC counts")
    plt.title("raw waveform")

    dir = os.path.dirname(__file__)
    plt.savefig(f'{dir}/test_event.pdf')
