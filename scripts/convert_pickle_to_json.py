"""
Specifically for creating scale parameter jsons to implement in the NuRadio wiki
"""


import argparse
from astropy.time import Time
import copy
import json
import numpy as np

from utilities.NpEncoder import NpEncoder
from utilities.utility_functions import read_pickle


def reduce_contents_size(contents, freq_samples=128):
    contents_cp = copy.copy(contents)
    del contents_cp["header"]["event_info"]
    del contents_cp["spec_amplitude_histograms"]

    step = int(len(contents["freq"])/freq_samples)

    contents_cp["freq"] = [contents["freq"][0]] + contents["freq"][1::step]
    for channel_id, _ in enumerate(contents["scale_parameters"]):
        contents_cp["scale_parameters"][channel_id] = [contents["scale_parameters"][channel_id][0]] + contents["scale_parameters"][channel_id][1::step]
        contents_cp["scale_parameters_cov"][channel_id] = [contents["scale_parameters_cov"][channel_id][0]] + contents["scale_parameters_cov"][channel_id][1::step]

    return contents_cp
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle")
    args = parser.parse_args()

    contents = read_pickle(args.pickle)
    print(contents.keys())
    if type(contents["time"]) is Time:
        contents["time"] = contents["time"].to_value("unix") 

    contents_reduced = reduce_contents_size(contents)
    print(contents_reduced.keys())
    print(np.array(contents_reduced["scale_parameters"]).shape)

    json_path = args.pickle.split(".", 1)[0]
    json_path += "_test.json"
    print(f"Saving as {json_path}")
    with open (json_path, "w") as json_file:
        json.dump(contents_reduced, json_file, cls=NpEncoder)