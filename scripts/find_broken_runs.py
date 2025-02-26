import os
import glob
import pickle
import argparse
import mattak.Dataset




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "%(prog)s",
                                     usage = "placeholder")
    parser.add_argument("-d", "--data_dir",
                        default = None)
    parser.add_argument("-s", "--station",
                        type = int,
                        default = 24)
    parser.add_argument("-r", "--run",
                        default = None)
    parser.add_argument("--debug", action = "store_true")
    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir

    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*") # run 363 is broken (100 waveforms with 200 event infos)
    print(root_dirs)

    mattak_kw = dict(backend = "pyroot", read_daq_status = False)
    broken_runs = {}
    try:
        i = 0
        for root_dir in root_dirs:
            print(f"at {i}/{len(root_dirs)}")
            i += 1
            try:
                ds = mattak.Dataset.Dataset(station = 0, run = 0, data_path=root_dir, backend = "pyroot")
            except (ValueError, OSError, ReferenceError, FileNotFoundError, NameError) as error:
                print(f"{error} for {root_dir}")
                run_nr = root_dir.split("/")[-1].split("run")[-1]
                broken_runs[str(run_nr)] = [root_dir, error]
                continue
    except KeyboardInterrupt:
        print(f"Sweep interupted")
    print(broken_runs)
    with open(f"/home/ulb/iihe/rcamphyn/noise_study/broken_runs/station{args.station}.pickle", "wb") as f:
        pickle.dump(broken_runs, f)
