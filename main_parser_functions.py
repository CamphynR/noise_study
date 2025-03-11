import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess

import NuRadioReco
import modules

from utilities.utility_functions import create_nested_dir
from utilities.NpEncoder import NpEncoder



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
        for cleaning_key in cleaning_modules.keys():
            cleaning_modules[cleaning_key].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])
    
        
    # construct data folder structure, the function that does the calculations should save the output here
    directory = construct_folder_hierarchy(config, args, folder_appendix)

    # actual calculations take place here
    # define variable options in kwargs in config file
    # define general behaviour in the calculation_function
    function_kwargs = config["variable_function_kwargs"][clean]
    calculated_output = calculation_function(reader, detector, config, args, logger, directory, cleaning_modules, **function_kwargs)

    # copies the config used for future reference
    config_name = f"{directory}/config.json"
    settings_dict = {**config, **vars(args)}
    if os.path.isfile(config_name):
        os.remove(config_name)
    with open(config_name, "w") as f:
        json.dump(settings_dict, f, cls=NpEncoder)

    if config["save"]:
        src_dir = "/tmp/data/"
        dest_dir = config["save_dir"]

        # capture to stdout to use in pipeline 
        job_dir = directory.split("/")
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
    save_dir = "/tmp/data"
    if "simulation" in config.keys():
        if config["simulation"] == True:
            save_dir += "/simulations"
    else:
        save_dir += "/data"
    # defining a savename overwrites the data structure and simply saves everything in a file,
    # useful when running one script for multiple stations
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
        os.makedirs(station_dir)


    return directory



def calculate_average_fft(reader, detector, config, args, logger, directory, cleaning_modules):
    station_id = args.station
    station_info = detector.get_station(station_id)
    nr_channels = len(station_info["channels"])
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    average_frequency_spectrum = np.zeros((nr_channels, len(frequencies)))
    squared_frequency_spectrum = np.zeros((nr_channels, len(frequencies)))
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
            average_frequency_spectrum[channel_id] += spectrum
            squared_frequency_spectrum[channel_id] += spectrum**2

            if args.test:
                if nr_events == 1:
                    if channel_id == 0:
                        plt.plot(frequencies, spectrum)
                        plt.xlabel("freq / GHz")
                        plt.ylabel("spec a / V/GHz")
                        plt.savefig("test_ft")
                        plt.close()
                        trace = channel.get_trace()
                        plt.plot(trace)
                        plt.xlabel("samples")
                        plt.ylabel("amplitude / V")
                        plt.savefig("test_trace")
                        plt.close()


    average_frequency_spectrum /= nr_events
    squared_frequency_spectrum /= nr_events
    var_frequency_spectrum = squared_frequency_spectrum - average_frequency_spectrum**2
    
    header = {"nr_events" : nr_events}
    result_dict = {"header" : header,
                   "time" : station.get_station_time(),
                   "freq" : frequencies,
                   "frequency_spectrum" : average_frequency_spectrum,
                   "var_frequency_spectrum" : var_frequency_spectrum}

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/average_ft"
        if args.batch_i is not None:
            filename += f"_batch{args.batch_i}"
        filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result_dict, f)

def populate_spec_amplitude_histogram(reader, detector, config, args, logger, directory, cleaning_modules, hist_range, nr_bins):
    # this assumes these parameters stay constant over the chosen runtime!
    station_id = args.station
    station_info = detector.get_station(station_id)
    nr_channels = len(station_info["channels"])
    # assumes the same nr of smaples and sampling rate between channels
    # assumtion due to sampling parameters being stored on the station_info level
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    
    clean_data = not args.skip_clean
    clean = "raw" if args.skip_clean else "clean"

    times = [time["readoutTime"] for time in reader.reader.get_events_information("readoutTime").values()]
    begin_time = np.min(times)
    end_time = np.max(times)
    
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

            if args.test and nr_events == 1:
                test_idx = 250
                plt.plot(frequencies, spectrum)
                plt.hlines(bin_edges, 0, 1.2*np.max(frequencies), colors="r", linewidth=1.)
                plt.scatter(frequencies[test_idx], spectrum[test_idx], color="red", zorder=5)
                x_text = [1.75 for i in range(len(bin_centres))]
                y_text = [bin_c for bin_c in bin_centres]
                text = [f"bin {i}" for i in range(len(bin_centres))]
                for i in range(len(text)):
                    plt.text(x_text[i], y_text[i], text[i], fontsize="xx-small",
                             verticalalignment="center",
                             horizontalalignment="center")
                plt.ylim(0.2, 0.4)
                plt.xlabel("freq / GHz")
                plt.ylabel("spec amp / V/GHz")
                plt.savefig("test_main")
                plt.close()
                logger.debug(f"test at frequency: {frequencies[test_idx]}")
                logger.debug(f"test binned in bin {bin_indices[test_idx]}, located at {bin_centres[bin_indices[test_idx]]}")

                raise OSError


    event_info = reader.reader.get_events_information(["run", "eventNumber"])
    header = {"nr_events" : nr_events,
              "hist_range" : hist_range,
              "bin_centres" : bin_centres,
              "begin_time" : begin_time,
              "end_time" : end_time,
              "events_info" : event_info}
    result_dict = {"header" : header,
                   "time" : station.get_station_time(),
                   "freq" : frequencies,
                   "spec_amplitude_histograms" : spec_amplitude_histograms}

    if config["save"]:
        filename = f"{directory}/station{station_id}/{clean}/spec_amplitude_histograms"
        if args.batch_i is not None:
            filename += f"_batch{args.batch_i}"
        filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result_dict, f)

    return result_dict    