import argparse
import matplotlib.pyplot as plt
import numpy as np
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs = "+")
    args = parser.parse_args()

    station_id = args.runs[0].split("/")[-2].split("station")[-1]
    run_nrs = []
    flower_gain_codes = []
    for run in args.runs:
        run_nr = os.path.basename(run).split("run")[-1]
        flower_gain_codes_path = os.path.join(run, "aux/flower_gain_codes.0.txt")
        with open(flower_gain_codes_path, "r") as f:
            codes = f.readlines()[1]
            codes = [int(c) for c in codes.split(" ")]
        run_nrs.append(run_nr)
        flower_gain_codes.append(codes)

    flower_gain_codes = np.array(flower_gain_codes)

    nr_channels = len(flower_gain_codes[0])

    plt.style.use("retro")
    fig, ax = plt.subplots()
    for channel_id in range(nr_channels):
        ax.plot(run_nrs, flower_gain_codes[:, channel_id], label=f"channel {channel_id}")
    ax.legend()
    ax.set_xticks(run_nrs[::20], labels=run_nrs[::20], rotation=45)
    ax.set_xlabel("run nr")
    ax.set_ylabel("gain code")
    fig.tight_layout()
    fig.savefig(f"figures/tests/flower_gain_codes_st{station_id}")
