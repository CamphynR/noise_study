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
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

#from dbAmplifier import dbAmplifier
from channelThermalNoiseAdder import channelThermalNoiseAdder
from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir
from temp_to_noise import temp_to_volt





def create_thermal_noise_events(nr_events, station_id, detector,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                noise_sources=["ice", "electronic"],
                                passband = None,
                                temperature_file=None):

    # detector and trace parameters
    station_info = detector.get_station(station_id)
    channel_ids = sorted([int(c) for c in station_info["channels"].keys()])
    if choose_channels is not None:
        channel_ids = choose_channels 
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    # electronic noise temperature
    temperature = 130 * units.kelvin
    # arbitrary choice but should be wide enough to incorporate detectors frequency fov
    min_freq = 10 * units.MHz
    max_freq = 1600 * units.MHz
    resistance = 50 * units.ohm
    amplitude = temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance,
                             filter_type="rectangular")

    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin(temperature_file)

    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()

    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()

#    db_amplifier = dbAmplifier(db=55, reduce=False)

    events = []
    for _ in range(nr_events):
        print(_)
        event = Event(run_number=-1, event_id=-1)
        event_electronic = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)
        event_electronic.set_station(station)

        
        if "ice" in noise_sources:
            thermal_noise_adder.run(event, station, detector, passband= [min_freq, max_freq])
        if "electronic" in noise_sources:
            generic_noise_adder.run(event_electronic, station, detector, amplitude=amplitude, min_freq=min_freq, max_freq=max_freq, type="rayleigh")
        if "sum" in noise_sources:
            station = event.get_station()
            station_electronic = event_electronic.get_station()
            for channel_id in channel_ids:
                trace = event.get_trace(channel_id)
                trace_electronic = event_electronic.get_trace()
                summed_trace= trace + trace_electronic


        if include_det_signal_chain:
            print("including the detector response")
#            db_amplifier.run(event, station, detector)
            hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event)

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
    save_dir +=f"/job_{date}_old_antenna"
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
        logging.getLogger().setLevel(logging.WARNING)
    print(save_dir)

    station_id = args.station
    channels_to_include = [0, 4]


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

#    for channel_id in channels_to_include:
#        if channel_id in channel_types["VPol"]:
#            antenna_model = antenna_models["VPol"]
#            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)
#
#        elif channel_id in channel_types["HPol"]:
#            antenna_model = antenna_models["HPol"]
#            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)


    print(detector.get_antenna_model(23, 0))
    print(detector.get_antenna_model(23, 4))


    temperature_file = "/home/ruben/Documents/projects/RNO-G_noise_study/sim/library/eff_temperature_-100_ntheta100.json"

    event_writer = eventWriter()

    def events_process(batch):
        events = create_thermal_noise_events(events_per_batch, args.station, detector,
                                             choose_channels = channels_to_include,
                                             include_det_signal_chain=config["include_det_signal_chain"],temperature_file=temperature_file,
                                             passband=[10 * units.MHz, 1600 * units.MHz]
                                             )

        batch_dir = f"station{args.station}" + "/" + f"run{batch}"
        os.makedirs(save_dir + "/" + batch_dir, exist_ok=True)
        filename = f"events_batch{batch}"
        savename = save_dir + "/" + batch_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in events:
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

        for event in events:
            station = event.get_station()
            channel = station.get_channel(0)
            nr_samples = 2048
            sampling_rate = 3.2 * units.GHz
            frequencies = np.fft.rfftfreq(nr_samples, d = 1./sampling_rate)
            frequency_spectrum = channel.get_frequency_spectrum()
            plt.plot(frequencies, np.abs(frequency_spectrum))
            plt.xlabel("freq / GHz")
            plt.ylabel("Spectral amplitude / V/GHZ")
            plt.savefig("test.png")
