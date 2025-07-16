import copy
import datetime
import glob
import json
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pickle
import subprocess

import NuRadioReco
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units
import modules

from utilities.utility_functions import create_nested_dir
from utilities.NpEncoder import NpEncoder


def open_select_runs_list(select_runs_path):
    with open(select_runs_path, "r") as select_runs_file:
        select_runs_list = json.load(select_runs_file)
    return select_runs_list



def find_data_files(args, config):

    if args.data_dir is None:
        data_dir = os.environ["RNO_G_DATA"]
    else:
        data_dir = args.data_dir


    if args.run is not None:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/")
        run_files = glob.glob(f"{data_dir}/station{args.station}/run{args.run}/*")
        is_root = np.any([run_file.endswith(".root") for run_file in run_files])
        is_nur = np.any([run_file.endswith(".nur") for run_file in run_files])
        return root_dirs, is_root, is_nur, config

    if args.select_runs_path is not None:
        select_runs_list = open_select_runs_list(args.select_runs_path)
#        print(select_runs_list)
#        for run in select_runs_list:
#            print(run)
#            glob.glob(f"{data_dir}/station{args.station}/run{run}")[0]
        root_dirs = [glob.glob(f"{data_dir}/station{args.station}/run{run}")[0] for run in select_runs_list]
    else:
        root_dirs = glob.glob(f"{data_dir}/station{args.station}/run*")

    run_files = glob.glob(f"{data_dir}/station{args.station}/run**/*", recursive=True)
    if not len(run_files):
        run_files = glob.glob(f"{data_dir}/station{args.station}/clean/*")
    is_root = np.any([run_file.endswith(".root") for run_file in run_files])
    is_nur = np.any([run_file.endswith(".nur") for run_file in run_files])

    if is_root == is_nur:
        raise OSError("Either root and nur files are mixed or ")

    if is_root:
        root_dirs = sorted(root_dirs, key=lambda root_dir : int(os.path.basename(root_dir)[3:]))
        if args.nr_batches is not None:
            root_dirs = np.array(root_dirs)
            root_dirs = np.array_split(root_dirs, args.nr_batches)
        channels_to_include = list(np.arange(24))
        config["channels_to_include"] = channels_to_include
        config["simulation"] = False
    elif is_nur:
        sec_config_path = glob.glob(f"{data_dir}/station{args.station}/config*")[0]
        sec_config_path = glob.glob(f"{data_dir}/station{args.station}/config*")[0]
        with open(sec_config_path, "r") as sec_config_file:
            sec_config = json.load(sec_config_file)
        config["simulation"] = sec_config.get("simulation", True)

        if config["simulation"]:
            config.update(sec_config)
            noise_sources = config["noise_sources"]
            root_dirs_list = []
            for noise_source in noise_sources:
                root_dirs_tmp = [glob.glob(f"{root_dir}/events_{noise_source}_batch*")[0] for root_dir in root_dirs]
                root_dirs_list.append(root_dirs_tmp)
            if config["include_sum"]:
                root_dirs_tmp = [glob.glob(f"{root_dir}/events_batch*")[0] for root_dir in root_dirs]
                root_dirs_list.append(root_dirs_tmp)
        else:
            run_files = natsorted(run_files)
            root_dirs = run_files
            if args.nr_batches is not None:
                root_dirs = np.array(run_files)
                root_dirs = np.array_split(root_dirs, args.nr_batches)
            channels_to_include = list(np.arange(24))
            config["channels_to_include"] = channels_to_include
            
    else:
        raise TypeError("Data extension not recognized")


    return root_dirs, is_root, is_nur, config




def parse_data(reader, detector, config, args, logger,
               calculation_function,
               folder_appendix=None):
    """
    """
    # Options to clean, to add a cleaning module add it to this dictionary and specify it's options in the config file
    cleaning_options = {"channelBandpassFilter" : NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter,
                        "hardwareResponseIncorporator" : NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator,
                        "cwFilter" : modules.cwFilter.cwFilter,
                        "channelSinewaveSubtraction" : NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction}

    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"
    if clean_data:
        logger.debug("Setting up cleaning modules")
        # initialise cleaning modules selected in config
        cleaning_modules = dict((cf, cleaning_options[cf]()) for cf in config["cleaning"].keys())
        for cleaning_key in list(cleaning_modules.keys()):
            if cleaning_key == "channelSinewaveSubtraction":
                cleaning_modules[f"{cleaning_key}_sec"] = copy.copy(cleaning_modules[cleaning_key])
                cleaning_modules[cleaning_key].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])
                cleaning_modules[f"{cleaning_key}_sec"].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])
            else:
                cleaning_modules[cleaning_key].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])
    
        
    # construct data folder structure, the function that does the calculations should save the output here
    directory = construct_folder_hierarchy(config, args, folder_appendix)

    # actual calculations take place here
    # define variable options in kwargs in config file
    # define general behaviour in the calculation_function
    function_kwargs = config["variable_function_kwargs"][clean]
    calculated_output = calculation_function(reader, detector, config, args, logger, directory, cleaning_modules, **function_kwargs)

    # copies the config used for future reference
    config_name = f"{directory}/station{args.station}/config.json"
    settings_dict = {**config, **vars(args)}
    if os.path.isfile(config_name):
        os.remove(config_name)
    with open(config_name, "w") as f:
        json.dump(settings_dict, f, cls=NpEncoder)

    if config["save"]:
        src_dir = "/tmp/tmp_noise_study/"
        if args.test:
            src_dir = "/user/rcamphyn/tmp/tmp_noise_study/"
        dest_dir = config["save_dir"]

        # capture to stdout to use in pipeline 
        job_dir = directory.split("/")
        if args.test:
            job_dir = job_dir[5:]
        else:
            job_dir = job_dir[3:]
        job_dir = "/".join(job_dir)
        job_dir = dest_dir + "/" + job_dir
        print(job_dir)
        subprocess.call(["rsync", "-vuar", src_dir, dest_dir], stdout=subprocess.DEVNULL)

    return calculated_output



def construct_folder_hierarchy(config, args, folder_appendix=None):
    if not config["save"]:
        return

    # assumes function name to be calculate_*
    function_name = config["variable"]
    # first save to /tmp directory native to computer node and afterwards copy to pnfs
    save_dir = "/tmp/tmp_noise_study"
    if args.test:
        save_dir = "/user/rcamphyn/tmp/tmp_noise_study"
    if "simulation" in config.keys():
        if config["simulation"] == True:
            save_dir += "/simulations"
        else:
            save_dir += "/data"
    else:
        save_dir += "/data"

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    directory = f"{save_dir}/{function_name}/job_{date}"

    if folder_appendix is not None:
        directory += f"_{folder_appendix}"
    if args.test:
        directory += "_test"
    if args.filename_appendix:
        appendix = "_" + args.filename_appendix
        directory += appendix
    create_nested_dir(directory)

    # fill directory
    clean = "raw" if args.skip_clean else "clean"
    station_dir = f"{directory}/station{args.station}/{clean}"
    if not os.path.exists(station_dir):
        os.makedirs(station_dir, exist_ok=True)


    return directory



def calculate_average_fft(reader, detector, config, args, logger, directory, cleaning_modules, max_nr_events=100):
    """
    reader is an NuRadio eventReader object
    """
    station_id = args.station
    station_info = detector.get_station(station_id)
    # temperory until 2024 det is added to database
    date = datetime.date.fromisoformat(config["run_time_range"][0])
    sampl_rate_date = datetime.date.fromisoformat("2023-12-31")
    if date > sampl_rate_date :
        sampling_rate = 2.4 * units.GHz
    else:
        sampling_rate = station_info["sampling_rate"]
    nr_channels = len(station_info["channels"])
    nr_samples = station_info["number_of_samples"]

    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    average_frequency_spectrum = np.zeros((nr_channels, len(frequencies)))
    squared_frequency_spectrum = np.zeros((nr_channels, len(frequencies)))
    nr_events = 0
    for event in reader.run():
        station = event.get_station(args.station)
        if args.debug:
#            print(event.get_id())
            print(nr_events)

        if nr_events == 0:
            begin_time = station.get_station_time()
            end_time = station.get_station_time()
        if station.get_station_time() < begin_time:
            begin_time = station.get_station_time()
        if station.get_station_time() > end_time:
            end_time = station.get_station_time()
        
        if clean_data:
            for cleaning_key in cleaning_modules.keys():

                if cleaning_key.endswith("_sec"):
                    cleaning_key_stripped = cleaning_key.split("_")[0]
                    run_kw = config["cleaning"][cleaning_key_stripped]["run_kwargs"]
                else:
                    run_kw = config["cleaning"][cleaning_key]["run_kwargs"]

                cleaning_modules[cleaning_key].run(event, station, detector,
                                                   **run_kw)
        
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            ch_frequencies = channel.get_frequencies()
            assert np.all(frequencies == ch_frequencies)

            spectrum = channel.get_frequency_spectrum()
            spectrum = np.abs(spectrum)
            average_frequency_spectrum[channel_id] += spectrum
            squared_frequency_spectrum[channel_id] += spectrum**2

#            if args.debug:
#                if nr_events == 1:
#                    if channel_id == 19:
#                        plt.plot(frequencies, spectrum)
#                        plt.xlabel("freq / GHz")
#                        plt.ylabel("spec a / V/GHz")
#                        plt.savefig("test_ft")
#                        plt.close()
#                        trace = channel.get_trace()
#                        plt.plot(trace)
#                        plt.xlabel("samples")
#                        plt.ylabel("amplitude / V")
#                        plt.savefig("test_trace")
#                        plt.close()
        nr_events += 1
        if args.debug:
            if nr_events == max_nr_events:
                break


    average_frequency_spectrum /= nr_events
    squared_frequency_spectrum /= nr_events
    var_frequency_spectrum = squared_frequency_spectrum - average_frequency_spectrum**2

    if args.debug:
        channel_id_test = 19
        plt.plot(frequencies, average_frequency_spectrum[channel_id_test])
        plt.xlabel("freq / GHz")
        plt.ylabel("spec a / V/GHz")
        plt.xlim(0., 1.)
        plt.savefig("figures/tests/test_average_spectrum")
        plt.close()
    
    header = {"nr_events" : nr_events,
              "begin_time" : begin_time,
              "end_time" : end_time}
    result_dict = {"header" : header,
                   "freq" : frequencies,
                   "frequency_spectrum" : average_frequency_spectrum,
                   "var_frequency_spectrum" : var_frequency_spectrum}

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/average_ft"
        if args.batch_i is not None:
            filename += f"_run{args.batch_i}"
        filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result_dict, f)

def populate_spec_amplitude_histogram(reader, detector, config, args, logger, directory, cleaning_modules, hist_range, nr_bins):
    # this assumes these parameters stay constant over the chosen runtime!
    station_id = args.station
    station_info = detector.get_station(station_id)
    nr_channels = len(station_info["channels"])
    
    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    try:
        times = [time["readoutTime"] for time in reader.reader.get_events_information("readoutTime").values()]
        begin_time = np.min(times)
        end_time = np.max(times)
    except AttributeError:
        #sims
        begin_time = -1
        end_time = -1
    
    if args.nr_batches is None:
        # very bruteforce way to get an idea of spectrum scale
        for event in reader.run():
            station = event.get_station()
            spec_max = 0
            for channel in station.iter_channels():
                if channel.get_id() not in config["channels_to_include"]:
                    continue
                frequencies = channel.get_frequencies()
                frequency_spectrum = channel.get_frequency_spectrum()
                spec_max_ch = np.max(np.abs(frequency_spectrum))
                if spec_max_ch > spec_max:
                    spec_max = spec_max_ch
            break 
        
        hist_range = [0, spec_max]
        config["hist_range"] = hist_range

    for event in reader.run():
        station = event.get_station()
        for channel in station.iter_channels():
            if channel.get_id() not in config["channels_to_include"]:
                continue
            sampling_rate = channel.get_sampling_rate()
            nr_samples = channel.get_number_of_samples()
            frequencies = channel.get_frequencies()
            break
        break 

    bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)
    bin_centres = bin_edges[:-1] + np.diff(bin_edges[0:2]) / 2

    
    # populate histogram per channel and per frequency
    # so shape of data structure storing histograms is (channels, frequencies, bins)
    spec_amplitude_histograms = np.zeros((nr_channels, len(frequencies), nr_bins))

    nr_events = 0
    for event in reader.run():
        nr_events += 1
        station = event.get_station(args.station)
        
        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector,
                                                   **config["cleaning"][cleaning_key]["run_kwargs"])
        
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            ch_frequencies = channel.get_frequencies()
            assert np.all(frequencies == ch_frequencies)

            spectrum = channel.get_frequency_spectrum()
            spectrum = np.abs(spectrum)

            # to get a bin number you should only pass the INNER edges to np.searchsorted
            bin_indices = np.searchsorted(bin_edges[1:-1], spectrum)
            spec_amplitude_histograms[channel_id, np.arange(len(frequencies)), bin_indices] += 1

            # if args.test and nr_events == 1:
            #     test_idx = 250
            #     plt.plot(frequencies, spectrum)
            #     plt.hlines(bin_edges, 0, 1.2*np.max(frequencies), colors="r", linewidth=1.)
            #     plt.scatter(frequencies[test_idx], spectrum[test_idx], color="red", zorder=5)
            #     x_text = [1.75 for i in range(len(bin_centres))]
            #     y_text = [bin_c for bin_c in bin_centres]
            #     text = [f"bin {i}" for i in range(len(bin_centres))]
            #     for i in range(len(text)):
            #         plt.text(x_text[i], y_text[i], text[i], fontsize="xx-small",
            #                  verticalalignment="center",
            #                  horizontalalignment="center")
            #     plt.ylim(0.2, 0.4)
            #     plt.xlabel("freq / GHz")
            #     plt.ylabel("spec amp / V/GHz")
            #     plt.savefig("test_main")
            #     plt.close()
            #     logger.debug(f"test at frequency: {frequencies[test_idx]}")
            #     logger.debug(f"test binned in bin {bin_indices[test_idx]}, located at {bin_centres[bin_indices[test_idx]]}")

            #     raise OSError


    try:
        event_info = reader.reader.get_events_information(["run", "eventNumber"])
    except AttributeError:
        event_info = 0
    header = {"nr_events" : nr_events,
              "hist_range" : hist_range,
              "bin_centres" : bin_centres,
              "begin_time" : begin_time,
              "end_time" : end_time,
              "events_info" : event_info,
              "sampling_rate" : sampling_rate,
              "nr_samples" : nr_samples}
    result_dict = {"header" : header,
                   "time" : station.get_station_time(),
                   "freq" : frequencies,
                   "spec_amplitude_histograms" : spec_amplitude_histograms}

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/spec_amplitude_histograms"
        if args.batch_i is not None:
            filename += f"_run{args.batch_i}"
        filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result_dict, f)

    return result_dict    



def collect_traces(reader, detector, config, args, logger, directory, cleaning_modules, max_nr_events=1000):
    station_id = args.station
    station_info = detector.get_station(station_id)
    nr_channels = len(station_info["channels"])
    nr_samples = station_info["number_of_samples"]

    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    traces = []
    nr_events = 0
    for event in reader.run():
        nr_events += 1
        station = event.get_station(args.station)
        
        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector,
                                                   **config["cleaning"][cleaning_key]["run_kwargs"])
        trace_ch = []
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            trace = channel.get_trace()
            trace_ch.append(trace)
        traces.append(trace_ch)
        if nr_events == max_nr_events:
            break
    
    traces = np.array(traces)

    
    header = {"nr_events" : nr_events}
    result_dict = {"header" : header,
                   "time" : station.get_station_time(),
                   "traces" : traces}

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/traces"
        if args.batch_i is not None:
            filename += f"_batch{args.batch_i}"
        filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result_dict, f)


def collect_spectra(reader, detector, config, args, logger, directory, cleaning_modules):
    """
    Purely for use on DATA files!
    """
    station_id = args.station
    station_info = detector.get_station(station_id)
    nr_channels = len(station_info["channels"])
    nr_samples = station_info["number_of_samples"]

    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    config["simulation"] = False

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/spectra"
        if args.batch_i is not None:
            filename += f"_run{args.batch_i}"
        filename += ".nur"
        event_writer = eventWriter()
        event_writer.begin(filename)

    for event in reader.run():
        station = event.get_station(args.station)
        
        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector,
                                                   **config["cleaning"][cleaning_key]["run_kwargs"])
        for channel in station.iter_channels():
            # forces NuRadio to store spectra instead of traces
            spectrum = channel.get_frequency_spectrum()
        event_writer.run(event, None, mode={"Channels": True})
    event_writer.end()



functions = dict(average_ft = calculate_average_fft,
                 spec_hist = populate_spec_amplitude_histogram,
                 traces = collect_traces,
                 spectra = collect_spectra)
