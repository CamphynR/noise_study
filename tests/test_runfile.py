import argparse
import uproot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d")
    args = parser.parse_args()

    with uproot.open(f"{args.data}/waveforms.root") as f:
        print(f.keys())
        print(f["waveforms"].keys())
        print(f["waveforms/waveforms"].arrays(library="np"))
