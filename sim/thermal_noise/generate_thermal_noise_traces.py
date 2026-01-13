import argparse
from astropy.time import Time
import datetime
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import subprocess

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
#from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities.signal_processing import calculate_vrms_from_temperature

from dbAmplifier import dbAmplifier
from modules.channelThermalNoiseAdder import channelThermalNoiseAdder
from modules.channelGalacticSunNoiseAdder import channelGalacticSunNoiseAdder
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator

from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir
from temp_to_noise import temp_to_volt


from modules.channelNoiseFromTemperatureAdder import channelNoiseFromTemperatureAdder


def padding_function(trace):
    padded_trace = np.pad(trace, len(trace), mode="reflect")
    filt = np.r_[np.linspace(0., 1., len(trace)/2), np.ones_like(trace), np.linspace(1., 0., len(trace)/2)]
    padded_trace *= filt
    return padded_trace


def create_sim_event(station_id, channel_ids, detector, frequencies, sampling_rate):
    event = Event(run_number=-1, event_id=-1)
    station = Station(station_id)
    station.set_station_time(detector.get_detector_time())
    for channel_id in channel_ids:
        channel = Channel(channel_id)
        channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
        station.add_channel(channel)
    event.set_station(station)
    return event


def get_traces_from_event(event):
    station = event.get_station()
    traces = []
    for channel in station.iter_channels():
        trace = channel.get_trace()
        traces.append(trace)
    return np.array(traces)



def create_thermal_noise_events(nr_events, station_id, detector,
                                config,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                noise_sources=["ice", "electronic", "galactic"],
                                include_sum = True,
                                electronic_temperature=80*units.kelvin,
                                passband = None,
                                padding_length=8192,
                                use_s_param_hardware=False,
                                debug=False):
    # electronic noise temperature, refer to eric's POS for this (PoS(ICRC2023)1171)

    # detector and trace parameters
    station_info = detector.get_station(station_id)
    channel_ids = np.arange(24)
    if choose_channels is not None:
        channel_ids = choose_channels 

#    # need to still implement trigger option for get_number_of_samples
#    trigger = config["digitizer"] == "flower"
#    nr_samples = detector.get_number_of_samples(station_id, 0, trigger=trigger)
#    sampling_rate = detector.get_sampling_frequency(station_id, 0, trigger=trigger)

    # temporary untill 2024 detector is included in the database
    digitizer_config_path = "configs/digitizer_settings.json"
    digitizer_settings = read_config(digitizer_config_path)
    nr_samples = digitizer_settings[config["digitizer"]]["nr_samples"]
    sampling_rate = digitizer_settings[config["digitizer"]]["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    electronic_noise_type = config["electronic_noise_type"]


    # arbitrary choice but should be wide enough to incorporate detectors frequency fov
    min_freq = 10 * units.MHz
    max_freq = 1600 * units.MHz
    resistance = 50 * units.ohm
    if isinstance(electronic_temperature, dict):
        amplitude = {}
        for key in electronic_temperature.keys():
            amplitude[key] = calculate_vrms_from_temperature(electronic_temperature[key],
                                                             bandwidth=[min_freq, max_freq],
                                                             impedance=resistance)
    else:
        amplitude = calculate_vrms_from_temperature(electronic_temperature, bandwidth=[min_freq, max_freq], impedance=resistance)


    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin(sim_library_dir=config["sim_library_dir"],
                              debug=debug)

    if electronic_noise_type == "flat_temp":
        generic_noise_adder = channelGenericNoiseAdder()
        generic_noise_adder.begin()
        electronic_noise_kwargs = dict(amplitude=amplitude, min_freq=min_freq, max_freq=max_freq, type="rayleigh")
    elif electronic_noise_type == "measurement":
        eletronic_noise_measurements = config["electronic_noise_measurements"]
        generic_noise_adder = channelNoiseFromTemperatureAdder()
        generic_noise_adder.begin(eletronic_noise_measurements)
        electronic_noise_kwargs = dict()

    galactic_noise_adder = channelGalacticNoiseAdder()
    galactic_noise_adder.begin(freq_range=[min_freq, max_freq],
                               caching=True)

    if use_s_param_hardware:
        system_response = hardwareResponseIncorporator()
        system_response.begin()
        system_response_kwargs = {sim_to_data : True}

    else:
        system_response = systemResponseTimeDomainIncorporator()
        if config["digitizer"] == "radiant_v2":
            response_path = "sim/library/deep_impulse_responses.json"
        elif config["digitizer"] == "radiant_v3":
            response_path = ["sim/library/v2_v3_deep_impulse_responses.json", "sim/library/v2_v3_surface_impulse_responses.json"]
        elif config["digitizer"] == "flower":
            response_path = "sim/library/flower_impulse_responses.json"
        else:
            raise KeyError(f"config['digitizer'] is not a recognized season.")

        system_response.begin(det=detector,
                              response_path=response_path
                              )
        system_response_kwargs = {}


    events = []
    for _ in range(nr_events):
        if debug:
            print(f"on event {_}")
        nr_event_types = len(noise_sources)
        if include_sum:
            nr_event_types += 1
        event_types = [create_sim_event(station_id, channel_ids, detector,
                                        frequencies, sampling_rate) 
                       for event_type in range(nr_event_types)]
        
        for i, noise_source in enumerate(noise_sources):
            station = event_types[i].get_station()
            if noise_source == "ice":
                thermal_noise_adder.run(event_types[i], station, detector, passband= [min_freq, max_freq])
            elif noise_source == "electronic":
                generic_noise_adder.run(event_types[i], station, detector, **electronic_noise_kwargs)
            elif noise_source == "galactic":
                galactic_noise_adder.run(event_types[i], station, detector)
            elif noise_source == "galactic_min":
                time = datetime.datetime(2023, 8, 13, 11)
                galactic_noise_adder.run(event_types[i], station, detector, manual_time=time)
            elif noise_source == "galactic_max":
                time = datetime.datetime(2023, 8, 13, 23)
                galactic_noise_adder.run(event_types[i], station, detector, manual_time=time)

        if include_sum:
            # traces_sum = np.zeros((len(channel_ids), nr_samples))
            traces_sum = np.zeros((len(channel_ids), nr_samples + padding_length))
            for event in event_types:
                traces = get_traces_from_event(event)
                traces_padded = np.array(
                    [np.pad(trace, int(padding_length/2), mode="constant", constant_values=0.) for trace in traces])
                traces_sum += traces_padded
            station = Station(station_id)
            station.set_station_time(detector.get_detector_time())
            for i, channel_id in enumerate(channel_ids):
                channel = Channel(channel_id)
                channel.set_trace(traces_sum[i], sampling_rate)
                station.add_channel(channel)
            event_types[-1].set_station(station)


        if include_det_signal_chain:
            for event in event_types:
                station = event.get_station()
                system_response.run(event, station, detector, **system_response_kwargs)
        events.append(event_types)
    
    events = np.array(events)
    # switch nr_events and noise_source indices
    events = events.T
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", default=23, type=int)
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_i", default=None)
    parser.add_argument("--name_appendix", default=None)
    args = parser.parse_args()

    config = read_config(args.config)

    if not args.debug:
        save_dir = f"/tmp/simulations/thermal_noise_traces"
    else:
        save_dir = f"/user/rcamphyn/tmp/simulations/thermal_noise_traces"
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}"
    if args.debug:
        save_dir += "_test"
    if config["use_s_param_hardware_response"]:
        save_dir += "s_param_hardware_response"
    if args.name_appendix is not None:
        save_dir += args.name_appendix

    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    settings_dict["simulation"] = True
    config_file = f"{save_dir}/station{args.station}/{os.path.basename(args.config)}"
    os.makedirs(f"{save_dir}/station{args.station}", exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(settings_dict, f, indent=4)
    
    if args.debug:
        log_level = logging.WARNING
    else:
        log_level = logging.CRITICAL
    
    logger = logging.getLogger("NuRadioReco")
    logger.setLevel(log_level)

    station_id = args.station
    channels_to_include = config["channels_to_include"]


    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPol" : [4, 8, 11, 21],
                     "LPDA" : [12, 13, 14, 15, 16, 17, 18, 19, 20]}
#    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
#                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74",
#                      "LPDA" : "createLPDA_100MHz_InfFirn_n1.4"}
    antenna_models = config["antenna_models"]

    logger.debug("querying detector")
    detector_time = Time(f"{config['season']}-8-1")


    json_filename = f"/user/rcamphyn/software/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_{config['season']}.json"
    with open(json_filename, "r") as json_file:
        det_dict = json.load(json_file)
        for key in det_dict["channels"].keys():
            if det_dict["channels"][key]["channel_id"] in channel_types["VPol"]:
                det_dict["channels"][key]["ant_type"] = antenna_models["VPol"]
            if det_dict["channels"][key]["channel_id"] in channel_types["HPol"]:
                det_dict["channels"][key]["ant_type"] = antenna_models["HPol"]
            if det_dict["channels"][key]["channel_id"] in channel_types["LPDA"]:
                det_dict["channels"][key]["ant_type"] = antenna_models["LPDA"]

    # when using a dict like this you have to deserialize the jsonbecause TinyDB already serialized the dates to strings
    for station in det_dict["stations"].values():
        for key, val in station.items():
            if "{TinyDate}" in str(val):
                station[key] = Time(val.split(":", 1)[1], format="isot")

    for channel in det_dict["channels"].values():
        for key, val in channel.items():
            if "{TinyDate}" in str(val):
                channel[key] = Time(val.split(":", 1)[1], format="isot")

    detector = Detector(dictionary=det_dict, source="dictionary", antenna_by_depth=False)
    detector.update(detector_time)
    logger.debug("done querying detector")


# For use when 2024 is implemented in DB
#    detector = ModDetector(log_level=log_level,
#                           select_stations=station_id,
#                            )
#                           database_time=detector_time)

#    for channel_id in channels_to_include:
#        if channel_id in channel_types["VPol"]:
#            antenna_model = antenna_models["VPol"]
#            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)
#
#        elif channel_id in channel_types["HPol"]:
#            antenna_model = antenna_models["HPol"]
#            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)



    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    electronic_temperature = config["electronic_temperature"] * units.kelvin

    event_writer = eventWriter()

    def events_process(batch_i, debug=False):

        events = create_thermal_noise_events(events_per_batch, args.station, detector, config,
                                             choose_channels = channels_to_include,
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             noise_sources=noise_sources, include_sum=include_sum,
                                             electronic_temperature=electronic_temperature,
                                             passband=[10 * units.MHz, 1600 * units.MHz],
                                             use_s_param_hardware=config["use_s_param_hardware_response"],
                                             debug=debug
                                             )


        batch_dir = f"station{args.station}" + "/" + f"run{batch_i}"
        os.makedirs(save_dir + "/" + batch_dir, exist_ok=True)

        filenames = [f"events_{noise_source}_batch{batch_i}" for noise_source in noise_sources]
        if include_sum:
            filename = f"events_batch{batch_i}"
            filenames.append(filename)

        for i, filename in enumerate(filenames):
            savename = save_dir + "/" + batch_dir + "/" + filename
            event_writer.begin(filename=savename)
            for event in events[i]:
                event_writer.run(event)
        dest_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 

        # save to stdout to use as path in pipeline
        print(dest_dir + os.path.basename(save_dir))
        subprocess.call(["rsync", "-vuar", save_dir, dest_dir], stdout=subprocess.DEVNULL)
        

        if not args.debug:
            del events
        else:
            return events

    if not args.debug:
        if args.batch_i is None:
            nr_batches = 10
            events_per_batch = 200
            with multiprocessing.Pool() as p:
                p.map(events_process, range(nr_batches))
        else:
            events_per_batch = 200
            events_process(args.batch_i)




    if args.debug:
        nr_batches = 1
        events_per_batch = 200
        noise_sources = ["electronic"]
        events = events_process(0, debug=False)

        labels = ["ice"]
#        labels = ["ice", "electronic", "galactic", "sum"]
        fig, ax = plt.subplots(figsize=(20,10))
        freq_spectra = []

        for i, event in enumerate(events[0]):
            station = event.get_station()
            channel = station.get_channel(0)
            nr_samples = channel.get_number_of_samples()
            sampling_rate = channel.get_sampling_rate()
            frequencies = np.fft.rfftfreq(nr_samples, d = 1./sampling_rate)
            frequency_spectrum = channel.get_frequency_spectrum()
            freq_spectra.append(frequency_spectrum)
        frequency_spectrum_mean = np.mean(np.abs(freq_spectra), axis=0)

        ax.plot(frequencies, frequency_spectrum_mean)
        ax.set_xlabel("freq / GHz")
        ax.set_ylabel("Spectral amplitude / V/GHZ")
        ax.set_xlim(None, 1.)
        fig.savefig("test_sim.png")
