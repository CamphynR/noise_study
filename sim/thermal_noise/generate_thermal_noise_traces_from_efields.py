import argparse
import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from NuRadioReco.detector import detector
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelThermalNoiseAdder import channelThermalNoiseAdder
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

from dbAmplifier import dbAmplifier
from channelThermalNoiseNoAntennaAdder import channelThermalNoiseNoAntennaAdder
from utilities.utility_functions import read_pickle, write_pickle, read_config, create_nested_dir
from temp_to_noise import temp_to_volt





def create_thermal_noise_events(nr_events, station_id, detector,
                                choose_channels=None,
                                include_det_signal_chain=True,
                                noise_sources=["ice", "electronic"],
                                **thermal_noise_kwargs):

    # detector and trace parameters
    station_info = detector.get_station(station_id)
    channel_ids = sorted([int(c) for c in station_info["channels"].keys()])
    if choose_channels is not None:
        channel_ids = choose_channels 
    nr_samples = station_info["number_of_samples"]
    sampling_rate = station_info["sampling_rate"]
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)

    # electronic noise temperature
    temperature = 130*units.kelvin
    # arbitrary choice but should be wide enough to incorporate detectors frequency fov
    min_freq = 10 * units.MHz
    max_freq = 1600 * units.MHz
    resistance=50*units.ohm
    amplitude = temp_to_volt(temperature, min_freq, max_freq, frequencies, resistance,
                             filter_type="rectangular")

    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin(**thermal_noise_kwargs)

    generic_noise_adder = channelGenericNoiseAdder()
    generic_noise_adder.begin()

    hardware_response = hardwareResponseIncorporator()
    hardware_response.begin()

#    db_amplifier = dbAmplifier(db=55, reduce=False)

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

        if "ice" in noise_sources:
            thermal_noise_adder.run(event, station, detector, passband= [min_freq, max_freq])
        if "electronic" in noise_sources:
            generic_noise_adder.run(event, station, detector, amplitude=amplitude, min_freq=min_freq, max_freq=max_freq, type="rayleigh")

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
    args = parser.parse_args()

    config = read_config(args.config) 

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/config_efields.json"
    if os.path.exists(config_file):
        print("overwriting config file")
        os.remove(config_file)
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)

    logging.getLogger().setLevel(logging.DEBUG)
    
    print(save_dir)

    station_id = args.station
    include_channels = [0, 1, 2, 3, 4, 8]

    channel_mapping = {}

    nr_batches = 3
    events_per_batch = 200

    n_side = 4
    noise_temperature = 300 * units.kelvin

    def get_antenna_type(ch_id):
        if ch_id in [0, 1, 2, 3]:
            return "VPol"
        elif ch_id in [4, 8]:
            return "HPol"
        else:
            return "LPDA"

    antenna_models = {"HPol" : "RNOG_hpol_v4_8inch_center_n1.74",
                      "VPol" : "RNOG_vpol_v3_5inch_center_n1.74"}

        
    
#    detector = detector.Detector(source="rnog_mongo", select_stations=station_id)
#    detector_time = datetime.datetime(2022, 8, 1)
#    detector.update(detector_time)

    detector = ModDetector(database_connection='RNOG_public', log_level=logging.NOTSET, over_write_handset_values=None,
                 database_time=None, always_query_entire_description=False, detector_file=None,
                 select_stations=station_id, create_new=False)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

#    for channel_id in include_channels:
#        antenna_type = get_antenna_type(channel_id)
#        antenna_model = antenna_models[antenna_type]
#        detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)

    for channel_id in [0, 1, 2, 3]:
        antenna_model = "RNOG_vpol_v3_5inch_center_n1.74"
        detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)

    for channel_id in [4, 8]:
        antenna_model = "RNOG_hpol_v4_8inch_center_n1.74"
        detector.modify_channel_description(station_id, channel_id, ["signal_chain","VEL"], antenna_model)


    print(detector.get_antenna_model(23, 0))
    print(detector.get_antenna_model(23, 4))

    event_writer = eventWriter()

    print(config["include_det_signal_chain"])

    for batch in range(nr_batches): 
        ice_events = create_thermal_noise_events(events_per_batch, args.station, detector, choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             n_side=n_side, noise_temperature=noise_temperature, noise_sources=["ice"])
        electronic_events = create_thermal_noise_events(events_per_batch, args.station, detector, choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             n_side=n_side, noise_temperature=noise_temperature, noise_sources=["electronic"])
        events = create_thermal_noise_events(events_per_batch, args.station, detector, choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             n_side=n_side, noise_temperature=noise_temperature)

        filename = f"ice_events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in ice_events:
            event_writer.run(event)
        del ice_events

        filename = f"electronic_events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in electronic_events:
            event_writer.run(event)
        del electronic_events

        filename = f"events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in events:
            event_writer.run(event)
        del events

#    test_file = "./testing_sims.pickle"
#    events = read_pickle(test_file)

#    freq = 400 * units.MHz
#    channel_id = 0
#    station = events[0].get_station(station_id)
#    channel = station.get_channel(channel_id)
#    frequencies = channel.get_frequencies()
#    spectrum = channel.get_frequency_spectrum()
#    trace = channel.get_trace()
#
#    fig, ax = plt.subplots()
#    ax.plot(trace)
#    ax.set_xlabel("samples")
#    ax.set_ylabel("V")
#    fig.savefig("test_trace")
#
#
#    fig, ax = plt.subplots()
#    ax.plot(frequencies, np.abs(spectrum))
#    ax.set_xlabel("freq / GHz")
#    ax.set_ylabel("spectral ampltiude / V/GHz")
#    fig.savefig("test_freq")
#
#    hist_values = []
#    for event in events:
#        station = event.get_station()
#        channel = station.get_channel(channel_id)
#        frequencies = channel.get_frequencies()
#        spectrum = channel.get_frequency_spectrum()
#        hist_values.append(np.abs(spectrum[np.where(freq==frequencies)]))
#
#
#    hist_values = np.squeeze(hist_values)
#    fig, ax = plt.subplots()
#    ax.hist(hist_values, bins=20, rwidth=0.9)
#    ax.set_xlabel("spectral ampltiude / V/GHz")
#    fig.savefig("test_distr")
