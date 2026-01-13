import argparse
import os

from utilities.utility_functions import read_pickle, write_pickle


def read_vrms_hist_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    nr_events = header["nr_events"]
    bin_centres = header["bin_centres"]
    vrms_hist = result_dictionary["vrms_histograms"]
    return bin_centres, vrms_hist, nr_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickles", nargs="+")
    args = parser.parse_args()

    run_nrs = []

    bin_centres_prev, vrms_hist_prev, nr_events_prev = read_vrms_hist_file(args.pickles[0])
    nr_events_combined = nr_events_prev
    vrms_hist_combined = vrms_hist_prev
    for i, pickle in enumerate(args.pickles[1:]):
        bin_centres, vrms_hist, nr_events = read_vrms_hist_file(pickle)
        nr_events_combined = nr_events_combined + nr_events
        vrms_hist_combined = vrms_hist_combined + vrms_hist
        run_nr = int(os.path.basename(pickle).split("run")[-1].split(".pickle")[0])
        run_nrs.append(run_nr)

    result_dictionary = read_pickle(args.pickles[0])
    result_dictionary["header"]["nr_events"] = nr_events_combined
    result_dictionary["header"]["run_nrs"] = run_nrs
    result_dictionary["vrms_hist"] = vrms_hist_combined


    pickle_file = args.pickles[0].rsplit("_", 1)[0]
    pickle_file += "_combined.pickle"
    write_pickle(result_dictionary, pickle_file)
