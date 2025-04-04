"""
File to generate thermal noise from ice starting from the electric fields at the antenna.
This file does NOT include the antenna profiles and allows for skipping the detector's hardware response.
For testing purposes
"""
import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from NuRadioReco.detector import detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelThermalNoiseAdder import channelThermalNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

from channelThermalNoiseNoAntennaAdder import channelThermalNoiseNoAntennaAdder
from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir



def create_thermal_noise_events(nr_events, station_id, detector,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                **thermal_noise_kwargs):
    station_info = detector.get_station(station_id)
    channel_ids = sorted([int(c) for c in station_info["channels"].keys()])
    if choose_channels is not None:
        channel_ids = choose_channels 
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    thermal_noise_adder = channelThermalNoiseNoAntennaAdder()
    thermal_noise_adder.begin(**thermal_noise_kwargs)

    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()

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
        thermal_noise_adder.run(event, station, detector, passband= [10*units.MHz,1600*units.MHz])
        if include_det_signal_chain:
            hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event)

    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=23)
    parser.add_argument("--skip_det", action="store_true")
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    args = parser.parse_args()

    config = read_config(args.config) 

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}_no_ant"
    if args.skip_det:
        save_dir += "_no_det"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/config_efields.json"
    if os.path.exists(config_file):
        print("overwriting config file")
        os.remove(config_file)
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)

    
    print(save_dir)

    station_id = args.station

    nr_batches = 1
    events_per_batch = 1

    n_side = 4
    noise_temperature = 300 * units.kelvin
    
    detector = detector.Detector(source="rnog_mongo", select_stations=station_id)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    event_writer = eventWriter()


    for batch in range(nr_batches): 
        events = create_thermal_noise_events(events_per_batch, 23, detector, choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=not args.skip_det,
                                             n_side=n_side, noise_temperature=noise_temperature)

        filename = f"events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in events:
            event_writer.run(event)
        del events
