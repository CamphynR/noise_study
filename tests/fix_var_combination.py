import argparse
from utilities.utility_functions import read_pickle





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()


    pickle_contents = []
    for pickle in args.pickles:
        contents = read_pickle(pickle)
        print(contents.keys())
        print(contents["var_frequency_spectrum"][0][30:500]/contents["frequency_spectrum"][0][30:500])
