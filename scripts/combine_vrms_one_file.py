import argparse
import json
import numpy as np




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()


    pickle_files = np.array_split(args.pickles, args.batch_size)
