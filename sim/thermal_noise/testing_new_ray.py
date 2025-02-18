import argparse
import datetime
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
print(multiprocessing.cpu_count())
import numpy as np
import os
import sys

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from channelIceNoiseAdderNewRay import channelIceNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium

import dbAmplifier
from utilities.utility_functions import read_config, create_nested_dir, select_ice_model


def create_thermal_noise_events(nr_events, station_id, detector,
                                ice_model, attenuation_model,
                                passband,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                log_level=logging.NOTSET,
                                args = {}):
    station_info = detector.get_station(station_id)
    channel_ids = sorted([int(c) for c in station_info["channels"].keys()])
    if choose_channels is not None:
        channel_ids = choose_channels 
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    thermal_noise_adder = channelIceNoiseAdder()
    thermal_noise_adder.begin(ice_model=ice_model, attenuation_model=attenuation_model,
                              d_theta = 5 * units.degree,
                              n_r=50,
                              R = 2000 * units.m,
                              log_level=log_level)

    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()

    db_amplifier = dbAmplifier.dbAmplifier()

    events = []
    for _ in range(nr_events):
        print(_)
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)


        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        
        event.set_station(station)
        thermal_noise_adder.run(event, station, detector, passband=passband)
        # test
#        db_amplifier.run(event, station, detector)

        if include_det_signal_chain:
            print("including hardware")
            hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event)

    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_nr", type=int, default=0)
    parser.add_argument("--station", "-s", type=int, default=23)
    parser.add_argument("--skip_det", action="store_true")
    parser.add_argument("--config", default="sim/thermal_noise/config_iceadder.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)

    station_id = args.station
    log_level = logging.WARNING
    channels_to_include = [0, 4]
    config["channels_to_include"] = channels_to_include

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}_iceadder"
    if args.skip_det:
        save_dir += "no_det"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    if args.batch_nr == 0:
        config_file = f"{save_dir}/config_efields.json"
        if os.path.exists(config_file):
            print("overwriting config file")
            os.remove(config_file)
        with open(config_file, "w") as f:
            json.dump(settings_dict, f)
    print(save_dir)


    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPol" : [4, 8, 11, 21]}
    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74"}
    

    detector = ModDetector(database_connection='RNOG_public', select_stations=station_id, log_level=log_level)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    for channel_id in channels_to_include:
        if channel_id in channel_types["VPol"]:
            antenna_model = antenna_models["VPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)

        elif channel_id in channel_types["HPol"]:
            antenna_model = antenna_models["HPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)


    ice_model = select_ice_model(config)
    attenuation_model = config["propagation"]["attenuation_model"]

    nr_events = 25

    event_writer = eventWriter()

    events = create_thermal_noise_events(nr_events, args.station, detector,
                                         ice_model=ice_model, attenuation_model=attenuation_model,
                                         passband=[10 * units.MHz, 1600 * units.MHz],
                                         choose_channels = channels_to_include,
#                                         include_det_signal_chain= not args.skip_det,
                                         include_det_signal_chain= True,
                                         log_level=log_level,
                                         args=args)

    batch_dir = f"station{args.station}" + "/" + f"run{args.batch_nr}"
    os.makedirs(save_dir + "/" + batch_dir)
    filename = f"events_batch{args.batch_nr}"
    savename = save_dir + "/" + batch_dir + "/" + filename


    event_writer.begin(filename=savename)
    for event in events:
        event_writer.run(event)

    if args.debug:
        for event in events:
            station = event.get_station()
            channel = station.get_channel(0)
            sampling_rate = 3.2 * units.GHz
            freqs = np.fft.rfftfreq(2048, d=1./sampling_rate)
            spec = channel.get_frequency_spectrum()
            plt.plot(freqs, np.abs(spec))
            plt.savefig("test.png")
            print(spec)



#    nr_batches = 1
#    nr_events = 1
#    for batch in range(nr_batches):
#        events = create_thermal_noise_events(nr_events, args.station, detector,
#                                             ice_model=ice_model, attenuation_model=attenuation_model,
#                                             passband=[10 * units.MHz, 1600 * units.MHz],
#                                             choose_channels = channels_to_include,
#                                             include_det_signal_chain=args.skip_det,
#                                             log_level=log_level)
#        filename = f"events_batch{batch}"
#        savename = save_dir + "/" + filename
#        event_writer.begin(filename=savename)
#        for event in events:
#            event_writer.run(event)
#        del events



#    def events_process(batch):
#        events = create_thermal_noise_events(nr_events, args.station, detector,
#                                             ice_model=ice_model, attenuation_model=attenuation_model,
#                                             passband=[10 * units.MHz, 1600 * units.MHz],
#                                             choose_channels = [0, 1, 2, 3, 4, 8],
#                                             include_det_signal_chain=args.skip_det)
#        
#        filename = f"events_batch{batch}"
#        savename = save_dir + "/" + filename
#        event_writer.begin(filename=savename)
#        for event in events:
#            event_writer.run(event)
#        del events
#        return
#
#    with multiprocessing.Pool() as p:
#        p.map(events_process, range(nr_batches))
