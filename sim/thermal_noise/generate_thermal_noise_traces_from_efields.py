import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
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

    thermal_noise_adder = channelThermalNoiseAdder()
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
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    args = parser.parse_args()

    config = read_config(args.config) 

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/config_efields.json"
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)

    
    print(save_dir)

    station_id = args.station
    nr_batches = 3
    events_per_batch = 100

    n_side = 4
    noise_temperature = 300 * units.kelvin
    
    detector = detector.Detector(source="rnog_mongo", select_stations=station_id)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    event_writer = eventWriter()


    for batch in range(nr_batches): 
        events = create_thermal_noise_events(events_per_batch, 23, detector, choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=config["include_det_signal_chain"],
                                             n_side=n_side, noise_temperature=noise_temperature)

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
