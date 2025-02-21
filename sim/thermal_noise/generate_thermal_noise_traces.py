import argparse
import datetime
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

#from dbAmplifier import dbAmplifier
from channelThermalNoiseAdder import channelThermalNoiseAdder
from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir
from temp_to_noise import temp_to_volt


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
                                choose_channels=None,
                                include_det_signal_chain=True,
                                noise_sources=["ice", "electronic", "galactic"],
                                include_sum = True,
                                electronic_temperature=80*units.kelvin,
                                passband = None):
    # electronic noise temperature, refer to eric's POS for this (PoS(ICRC2023)1171)

    # detector and trace parameters
    station_info = detector.get_station(station_id)
    channel_ids = sorted([int(c) for c in station_info["channels"].keys()])
    if choose_channels is not None:
        channel_ids = choose_channels 
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)


    # arbitrary choice but should be wide enough to incorporate detectors frequency fov
    min_freq = 10 * units.MHz
    max_freq = 1600 * units.MHz
    resistance = 50 * units.ohm
    amplitude = temp_to_volt(electronic_temperature, min_freq, max_freq, frequencies, resistance,
                             filter_type="rectangular")

    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin()

    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()

    galactic_noise_adder = channelGalacticNoiseAdder()
    galactic_noise_adder.begin(freq_range=[min_freq, max_freq],
                               caching=True)

    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()

    events = []
    for _ in range(nr_events):
        print(_)
        nr_event_types = len(noise_sources)
        if include_sum:
            nr_event_types += 1
        event_types = [create_sim_event(station_id, channel_ids, detector,
                                        frequencies, sampling_rate) 
                       for event_type in range(nr_event_types)]
        print(len(event_types))
        print(event_types[0])
        
        for i, noise_source in enumerate(noise_sources):
            print(i)
            station = event_types[i].get_station()
            if noise_source == "ice":
                thermal_noise_adder.run(event_types[i], station, detector, passband= [min_freq, max_freq])
            elif noise_source == "electronic":
                generic_noise_adder.run(event_types[i], station, detector, amplitude=amplitude, min_freq=min_freq, max_freq=max_freq, type="rayleigh")
            elif noise_source == "galactic":
                galactic_noise_adder.run(event_types[i], station, detector)

        if include_sum:
            traces_sum = np.zeros((len(channel_ids), nr_samples))
            for event in event_types:
                traces = get_traces_from_event(event)
                traces_sum += traces
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
                print("including the detector response")
                hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event_types)
    
    events = np.array(events)
    # switch nr_events and noise_source indices
    events = events.T
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=23, type=int)
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}"
    if args.debug:
        save_dir += "_test"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/config_efields.json"
    if os.path.exists(config_file):
        print("overwriting config file")
        os.remove(config_file)
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)
    print(save_dir)

    station_id = args.station
    channels_to_include = config["channels_to_include"]


    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPol" : [4, 8, 11, 21]}
    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74"}

    print("querying detector")
    detector = ModDetector(database_connection='RNOG_public', log_level=logging.NOTSET,
                           select_stations=station_id)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)
    print("done querying detector")

    for channel_id in channels_to_include:
        if channel_id in channel_types["VPol"]:
            antenna_model = antenna_models["VPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)

        elif channel_id in channel_types["HPol"]:
            antenna_model = antenna_models["HPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)


    print(detector.get_antenna_model(23, 0))
    print(detector.get_antenna_model(23, 4))

    noise_sources = config["noise_sources"]
    include_sum = config["include_sum"]
    electronic_temperature = config["electronic_temperature"] * units.kelvin

    event_writer = eventWriter()

    def events_process(batch):

        events = create_thermal_noise_events(events_per_batch, args.station, detector,
                                             choose_channels = channels_to_include,
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             noise_sources=noise_sources, include_sum=include_sum,
                                             electronic_temperature=electronic_temperature,
                                             passband=[10 * units.MHz, 1600 * units.MHz]
                                             )

        print(events.shape)
        batch_dir = f"station{args.station}" + "/" + f"run{batch}"
        os.makedirs(save_dir + "/" + batch_dir, exist_ok=True)

        filenames = [f"events_{noise_source}_batch{batch}" for noise_source in noise_sources]
        if include_sum:
            filename = f"events_batch{batch}"
            filenames.append(filename)
        print(filenames)

        for i, filename in enumerate(filenames):
            savename = save_dir + "/" + batch_dir + "/" + filename
            event_writer.begin(filename=savename)
            for event in events[i]:
                event_writer.run(event)
        if not args.debug:
            del events
        else:
            return events

    if not args.debug:
        nr_batches = 10
        events_per_batch = 300
        with multiprocessing.Pool() as p:
            p.map(events_process, range(nr_batches))




    if args.debug:
        nr_batches = 1
        events_per_batch = 1
        events = events_process(0)

        labels = ["ice", "electronic", "galactic", "sum"]
        fig, ax = plt.subplots()
        for i, event in enumerate(events):
            event = event[0]
            station = event.get_station()
            channel = station.get_channel(7)
            nr_samples = 2048
            sampling_rate = 3.2 * units.GHz
            frequencies = np.fft.rfftfreq(nr_samples, d = 1./sampling_rate)
            frequency_spectrum = channel.get_frequency_spectrum()
            ax.plot(frequencies, np.abs(frequency_spectrum), label=labels[i], zorder=-1*i)
            ax.set_xlabel("freq / GHz")
            ax.set_ylabel("Spectral amplitude / V/GHZ")
            ax.legend()
        fig.savefig("test.png")
