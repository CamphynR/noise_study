import argparse
from astropy.time import Time
import datetime
import json
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelIceNoiseAdder import channelIceNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium

from utilities.utility_functions import read_config, create_nested_dir


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

    thermal_noise_adder = channelIceNoiseAdder()
    thermal_noise_adder.begin()

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
        thermal_noise_adder.run(event, station, detector, **thermal_noise_kwargs)
        if include_det_signal_chain:
            hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event)

    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int, default=23)
    parser.add_argument("--skip_det", action="store_true")
    parser.add_argument("--config", default="sim/thermal_noise/config_iceadder.json")
    args = parser.parse_args()
    config = read_config(args.config)


    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}_iceadder"
    if args.skip_det:
        save_dir += "no_det"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/config_efields.json"
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)

    
    print(save_dir)

    detector = Detector(source="rnog_mongo", select_stations=args.station)
    det_time = Time("2022-08-01")
    detector.update(det_time)

    ice = medium.greenland_simple()
    model_ice = "GL2"

    nr_batches = 2
    nr_events = 100

    for batch in range(nr_batches):
        events = create_thermal_noise_events(nr_events, args.station, detector,
                                             choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=args.skip_det,
                                             ice=ice, model_ice=model_ice,
                                             passband=[10 * units.MHz, 1600 * units.MHz]
                                             )
        
        filename = f"events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in events:
            event_writer.run(event)
        del events
