import glob
import matplotlib.pyplot as plt
import numpy as np


from utilities.utility_functions import read_pickle



if __name__ == "__main__":

    season = 2023
    station_id = 11
    channel_id = 0

    data_path = f = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season{season}/station{station_id}/clean"
    print(f"{data_path}/average_vrms_run*.pickle")
    vrms_files = glob.glob(f"{data_path}/average_vrms_run*.pickle")
    print(vrms_files)

    for vrms_file in vrms_files[:2]:
        vrms_dict = read_pickle(vrms_file)
        print(vrms_dict.keys())



