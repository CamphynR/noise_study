import glob




if __name__ == "__main__":
    season = 2023
    station_id = 23
    channel_id = 0


    vrms_folder = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/vrms/complete_vrms_sets_v0.1/season{season}/station{station_id}/clean"
    vrms_run_paths = glob.glob(f"{vrms_folder}/average_vrms_run*.pickle")

    
    batch_size = 10
    vrms_run_paths = np.array_split(vrms_run_paths, len(vrms_run_paths))
