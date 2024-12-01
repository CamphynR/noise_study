import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=24)
    parser.add_argument("--channel", "-c", type=int, default=0)
    parser.add_argument("--run", "-r", type=int, default=100)
    parser.add_argument("--event", "-e", type=int, default=6)
    args = parser.parse_args()

    reader = readRNOGData()
    run_dir = glob.glob(f"{os.environ['RNO_G_DATA']}/station{args.station}/run{args.run}")
    selectors = [lambda event_info : event_info.triggerType == "FORCE"]
    reader.begin(run_dir,
                 read_calibrated_data=False,
                 selectors=selectors,
                 max_trigger_rate=2*units.Hz,
                 convert_to_voltage=True)

    
    event = reader.get_event_by_index(args.event)
    station = event.get_station()
    channel = station.get_channel(args.channel)
    # freqs are given in GHz (NuRadio's base frequency unit)
    freq = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()
    print(spec.shape)

    fig, ax = plt.subplots()
    ax.plot(freq, np.abs(spec))
    fig.savefig(f"single_spec_s{args.station}_c{args.channel}_run{args.run}_e{args.event}")
