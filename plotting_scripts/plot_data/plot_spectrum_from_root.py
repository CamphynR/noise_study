import argparse
import matplotlib.pyplot as plt
import numpy as np


from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/pnfs/iihe/rno-g/data/handcarry")
    parser.add_argument("--run", type=int)
    parser.add_argument("--station", type=int)
    parser.add_argument("--channels", type=int, nargs="+")
    args = parser.parse_args()

    data_path = f"{args.data_path}/station{args.station}/run{args.run}"


    run_types = ["physics"]
    max_trigger_rate = 2 * units.Hz

    selectors = [lambda event_info : event_info.triggerType == "FORCE"]


    mattak_kwargs = {
        "backend" : "uproot",
        }


    rnog_reader = readRNOGData()
    rnog_reader.begin(data_path,
                      run_types=run_types,
                      selectors=selectors,
                      max_trigger_rate=max_trigger_rate,
                      mattak_kwargs=mattak_kwargs)

    plt.style.use("retro")
    fig, ax = plt.subplots()
    for event in rnog_reader.run():
        station = event.get_station()
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if channel_id in args.channels:
                freq = channel.get_frequencies()
                spectrum = channel.get_frequency_spectrum()
                ax.plot(freq, np.abs(spectrum), label=f"channel {channel_id}", lw=1.5)
        break
    ax.legend()
    fig.savefig("test")
