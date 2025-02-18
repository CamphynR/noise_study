


def parse_variables(reader, detector, config, args,
                    calculate_variable = calculate_trace,
                    inter_event_calculation = None):
    """
    More basic wrapper for parsing through data. This only calculates variables on an event per event basis,
    there is no option to combine calculations between events.
    """
    clean_data = not args.skip_clean
    # initialise cleaning modules
    cleaning_options = {"channelBandpassFilter" : NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter,
                        "hardwareResponseIncorporator" : NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator,
                        "cwFilter" : modules.cwFilter.cwFilter}
    cleaning_modules = dict((cf, cleaning_options[cf]()) for cf in config["cleaning"].keys())

    for cleaning_key in cleaning_modules.keys():
        cleaning_modules[cleaning_key].begin(**config["cleaning"][cleaning_key]["begin_kwargs"])

    logger.debug("Starting calculation")
    if config["only_mean"]:
        variables_list = initialise_variables_list(calculate_variable)
        squares_list = initialise_variables_list(calculate_variable)
    else:
        variables_list = []
    
    clean = "raw" if args.skip_clean else "clean"
    kwargs = config["variable_function_kwargs"][clean]

    # initialise data folder structure and copy config
    directory = construct_folder_hierarchy(config, args)

    t0 = time.time()

    events_processed = 0
    for event in reader.run():
        if events_processed == 0:
            prev_run_nr = event.get_run_number()
        events_processed += 1
        run_nr = event.get_run_number()
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        logger.debug(f"Trigger is: {station.get_triggers()}")
        logger.debug(f"Run number is {event.get_run_number()}")

        if prev_run_nr != run_nr:

            if config["save"]:
                logger.debug(f"saving since {run_nr} != {prev_run_nr}")
                filename = f"{directory}/station{station_id}/{clean}/run{prev_run_nr}"
                filename += ".pickle"
                print(f"Saving as {filename}")
                with open(filename, "wb") as f:
                    pickle.dump(dict(time=station.get_station_time(), var=variables_list), f)

                if config["only_mean"]:
                    variables_list = initialise_variables_list(calculate_variable)
                    squares_list = initialise_variables_list(calculate_variable)
                else:
                    variables_list.clear()

        # there should be a mechanism in the detector code which makes sure
        # not to reload the detector for the same time stamps
        station_time = station.get_station_time()
        det.update(station_time)

        if clean_data:
            for cleaning_key in cleaning_modules.keys():
                cleaning_modules[cleaning_key].run(event, station, detector, **config["cleaning"][cleaning_key]["run_kwargs"])
       
        var_channels_per_event = []
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if config["only_mean"]:
                variables_list[channel_id, :] += calculate_variable(channel, **kwargs)
                squares_list[channel_id, :] += calculate_variable(channel, **kwargs)**2
            else:
                var_channels_per_event.append(calculate_variable(channel, **kwargs))

        variables_list.append(var_channels_per_event)
        
        prev_run_nr = event.get_run_number()



    if config["save"]:
        logger.debug(f"saving since {run_nr} != {prev_run_nr}")
        filename = f"{directory}/station{station_id}/{clean}/run{prev_run_nr}"
        filename += ".pickle"
        print(f"Saving as {filename}")
        with open(filename, "wb") as f:
            pickle.dump(dict(time=station.get_station_time(), var=variables_list), f)
        
        src_dir = "/tmp/data/"
        dest_dir = config["save_dir"]
        subprocess.call(["rsync", "-vuar", src_dir, dest_dir])
        

    if config["only_mean"]:
        print(f"total events that passed filter {events_processed}")
        variables_list = variables_list/events_processed
        var_list = squares_list/events_processed - variables_list**2
    

    dt = time.time() - t0
    logger.debug(f"Main calculation loop takes {dt}")

    return variables_list
