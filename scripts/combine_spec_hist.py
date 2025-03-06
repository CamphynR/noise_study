import argparse

from utilities.utility_functions import read_pickle, write_pickle


def read_spec_hist_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    bin_centres = header["bin_centres"]
    spec_hist = result_dictionary["spec_amplitude_histograms"]
    return bin_centres, spec_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles")
    args = parser.parse_args()
    
    bin_centres_prev, spec_hist_prev = read_spec_hist_file(args.pickles[0])
    spec_hist_combined = spec_hist_prev
    for i, pickle in enumerate(args.pickles[1:]):
        bin_centres, spec_hist = read_spec_hist_file(pickle)
        spec_hist_combined += spec_hist

    result_dictionary = read_pickle(args.pickles[0])
    result_dictionary["spec_amplitude_histograms"] = spec_hist_combined

    pickle_file = args.pickles[0].rsplit("_", 1)[0]
    pickle_file += "_combined.pickle"
    write_pickle(result_dictionary, pickle_file)
