import argparse
import datetime
import functools
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import constants

from NuRadioReco.detector import antennapattern
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units

from temp_to_noise import temp_to_volt
from utilities.utility_functions import read_config, create_nested_dir



def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z



#def average_antenna_vel_over_direction(antenna_pattern,
#                                       freqs,
#                                       orientation = [0, 0, np.pi / 2, np.pi / 2]):
#        
#    zenith = np.linspace(0, 89/180 * np.pi) 
#    azimuth = np.linspace(0, 2*np.pi) 
#    vel = []
#    for azi, zen in itertools.product(azimuth, zenith):
#        VEL = antenna_pattern.get_antenna_response_vectorized(freqs, zen, azi, *orientation)
#        vel.append([np.abs(VEL["theta"]), np.abs(VEL["phi"])])
#
#    vel = np.array(vel)
#    vel = vel.reshape(len(azimuth), len(zenith), 2, len(freqs))
#
#    # very much possible this needs to be changed!!
#    vel = np.mean(vel, axis=(0,1), dtype=np.float32)
#
#    return vel




class thermalNoiseVoltageSimulator():
    """
    Class formalism to yield simulated thermal noise with antenna imprint
    """

    def __init__(self):
        pass

    def begin(self, temp, bandwidth,
              min_freq=10*units.MHz, max_freq=1600*units.MHz,
              antenna_models = { "VPol" : "RNOG_vpol_ice_upsample", "HPol" :"RNOG_hpol_v4_8inch_center_n1.74"},
              caching=True):

        """
        temp : int
            temperature in kelvin
        bandpass : list
            frequencies in GHz
        """
        self.detector = detector
        
        self.station_id = station_id
        self.station_info = detector.get_station(station_id)

        self.channel_ids = detector.get_channel_ids(station_id)
        self.channel_ids = sorted(self.channel_ids)

        self.sampling_rate = self.station_info["sampling_rate"]
        self.nr_samples = self.station_info["number_of_samples"]
        self.frequencies = np.fft.rfftfreq(self.nr_samples, d=1./self.sampling_rate)

        self.temp = temp
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resistance=50*units.ohm

        self.bandwidth = bandwidth 
        self.amplitude = temp_to_volt(self.temp, min_freq, max_freq, self.frequencies, self.resistance)

        self.channel_bandpass_filter = channelBandPassFilter()

        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self.antenna_models = antenna_models

        self.channel_map = {i : "VPol" for i in [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]}
        hpol_map = {i : "HPol" for i in [4, 8, 11, 21]}
        self.channel_map.update(hpol_map)

        self.__caching = caching

    @functools.lru_cache(maxsize=1024 * 32)
    def get_cached_antenna_response(self, antenna_pattern, zen, azi, *ant_orient):
        return antenna_pattern.get_antenna_response_vectorized(self.frequencies, zen, azi, *ant_orient)

    @functools.lru_cache(maxsize=1024 * 32)
    def get_cached_antenna_response_averaged(self, antenna_pattern, orientation):       return average_antenna_vel_over_direction(antenna_pattern, self.frequencies, orientation)


    def average_antenna_vel_over_direction(self, antenna_pattern,
                                           freqs,
                                           orientation = [0, 0, np.pi / 2, np.pi / 2]):
            
        zenith = np.linspace(0, 89/180 * np.pi) 
        azimuth = np.linspace(0, 2*np.pi) 
        vel = []
        for azi, zen in itertools.product(azimuth, zenith):
            VEL = self.get_cached_antenna_response(antenna_pattern, zen, azi, *orientation)
            vel.append([np.abs(VEL["theta"]), np.abs(VEL["phi"])])

        vel = np.array(vel)
        vel = vel.reshape(len(azimuth), len(zenith), 2, len(freqs))

        # very much possible this needs to be changed!!
        vel = np.mean(vel, axis=(0,1), dtype=np.float32)

        return vel

    def run(self, event, station, detector):
        """
        sim_samples: int
            nr of frequency spectra to be simulated
        """
        channelgenericnoiseadder = channelGenericNoiseAdder()
 

        noise_spectrum = channelgenericnoiseadder.bandlimited_noise(
                    self.min_freq,
                    self.max_freq,
                    self.nr_samples, self.sampling_rate, self.amplitude,
                    type="rayleigh", time_domain=False)


        for channel in station.iter_channels():
            channel_id = channel.get_id()
            antenna_type = self.channel_map[channel_id]
            antenna_pattern = self.antenna_provider.load_antenna_pattern(self.antenna_models[antenna_type])
            vel = self.average_antenna_vel_over_direction(antenna_pattern, self.frequencies)

            det_resp = self.detector.get_signal_chain_response(station.get_id(), channel_id)
            det_resp = det_resp(np.array(self.frequencies))

                
            noise_spectrum_with_detector = noise_spectrum * (vel[0] + vel[1])
            noise_spectrum_with_detector = noise_spectrum_with_detector * det_resp

            channel.set_frequency_spectrum(noise_spectrum_with_detector, "same") 

        
        return



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

    thermal_noise_adder = thermalNoiseVoltageSimulator()
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
        thermal_noise_adder.run(event, station, detector)
        if include_det_signal_chain:
            hardware_response.run(event, station, det=detector, sim_to_data=True)
        events.append(event)

    return events


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=23)
    parser.add_argument("--config", default="sim/thermal_noise/config_efields.json")
    parser.add_argument("--skip_det", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config) 

    save_dir = f"{config['save_dir']}/simulations/thermal_noise_traces" 
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    save_dir +=f"/job_{date}_voltages"
    if args.skip_det:
        save_dir += "_no_det"
    create_nested_dir(save_dir)
    settings_dict = {**config, **vars(args)}
    config_file = f"{save_dir}/station{args.station}/config_voltages.json"
    with open(config_file, "w") as f:
        json.dump(settings_dict, f)

    
    print(save_dir)

    station_id = args.station
    nr_batches = 3
    events_per_batch = 100

    n_side = 4
    noise_temperature = 300 * units.kelvin
    
    detector = Detector(source="rnog_mongo", select_stations=station_id)
    detector_time = datetime.datetime(2022, 8, 1)
    detector.update(detector_time)

    event_writer = eventWriter()


    for batch in range(nr_batches): 
        events = create_thermal_noise_events(events_per_batch, args.station, detector,
                                             choose_channels = [0, 1, 2, 3, 4, 8],
                                             include_det_signal_chain=not args.skip_det,
                                             temp=noise_temperature, bandwidth=1500*units.MHz)

        filename = f"events_batch{batch}"
        savename = save_dir + "/" + filename
        event_writer.begin(filename=savename)
        for event in events:
            event_writer.run(event)
        del events
