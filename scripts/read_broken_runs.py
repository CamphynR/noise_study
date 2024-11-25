import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        broken_runs = pickle.load(f)

    print(broken_runs.keys())
