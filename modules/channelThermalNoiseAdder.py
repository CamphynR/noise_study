import functools
import json
import logging
import numpy as np
import os
import scipy.constants
import scipy.interpolate
from scipy.interpolate import interp1d
from urllib.request import urlretrieve
import warnings

from NuRadioReco.utilities import units, ice, geometryUtilities
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_station
import NuRadioReco.detector.antennapattern

import astropy.coordinates

from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.channelThermalNoiseAdder')


def structure_pickle_antenna_pattern(antenna_pattern):
    t0 = datetime.datetime.now()
    max_freq = 1.6
    (   ori_theta,
        ori_phi,
        rot_theta,
        rot_phi,
        frequencies,
        thetas,
        phis,
        RVEL_phi,
        RVEL_theta,
    ) = antenna_pattern
    thetas_unique = np.unique(thetas)
    phis_unique = np.unique(phis)
    frequencies_unique = np.unique(frequencies)
    max_freq_index = frequencies_unique <= max_freq
    RVEL_theta = np.array(RVEL_theta)
    RVEL_phi = np.array(RVEL_phi)

    RVEL_theta_new = np.zeros((len(thetas_unique), len(phis_unique),
                               len(frequencies_unique[max_freq_index])),
                              dtype=np.complex64)
    RVEL_phi_new = np.zeros((len(thetas_unique),
                             len(phis_unique),
                             len(frequencies_unique[max_freq_index])),
                            dtype=np.complex64)

    for zen_idx, zen in enumerate(thetas_unique):
        for azi_idx, azi in enumerate(phis_unique):
            for freq_idx, freq in enumerate(frequencies_unique[max_freq_index]):
                indices = np.where((thetas == zen) & (phis == azi) & (freq == frequencies))
                RVEL_theta_tmp = RVEL_theta[indices][0]
                RVEL_phi_tmp = RVEL_phi[indices][0]
                RVEL_theta_new[zen_idx, azi_idx, freq_idx] = RVEL_theta_tmp
                RVEL_phi_new[zen_idx, azi_idx, freq_idx] = RVEL_phi_tmp
    t1 = datetime.datetime.now() 
    print(t1 - t0)
    exit()

    
    RVEL_theta = np.interp(freqs, frequencies[max_freq_index], RVEL_theta[max_freq_index])
    RVEL_phi = np.interp(freqs, frequencies[max_freq_index], RVEL_phi[max_freq_index])


    antenna_response["theta"] = RVEL_theta
    antenna_response["phi"] = RVEL_phi
    return tuple(antenna_pattern)



def make_hashable(antenna_pattern):
    for i, content in enumerate(antenna_pattern):
        try:
            antenna_pattern[i] = tuple(content)
        except:
            continue
    antenna_pattern = tuple(antenna_pattern)
    return antenna_pattern



def check_angle_spacing(angles):
    diff_angles = np.diff(angles)
    #remove expected spacings
    diff_angles = diff_angles[diff_angles!=-np.pi]
    diff_angles = diff_angles[diff_angles!=-2*np.pi]
    diff_angles = diff_angles[diff_angles!=0]

    assert np.all(np.isclose(diff_angles, diff_angles[0], atol=1e-10)), "antenna VEL theta and signal incoming direction was not sampled at equal intervals"



def get_antenna_response_from_pickle_content(antenna_pattern, freqs, zen, azi):
    t0 = datetime.datetime.now()
    max_freq = 1.6
    antenna_response = {}

    (   ori_theta,
        ori_phi,
        rot_theta,
        rot_phi,
        frequencies,
        thetas,
        phis,
        RVEL_phi,
        RVEL_theta,
    ) = antenna_pattern
    dtheta = np.diff(thetas)[0]
    dphi = np.diff(phis)
    dphi = dphi[dphi!=0]
    dphi = dphi[0]
    dfreqs = np.diff(frequencies)[0]
    t1 = datetime.datetime.now() 
    print(t1 - t0)
    exit()

    indices = np.where((np.isclose(thetas,zen,atol=dtheta/2.)) & (np.isclose(phis,azi,atol=dphi/2.)))
    RVEL_theta = np.array(RVEL_theta)[indices]
    RVEL_phi = np.array(RVEL_phi)[indices]
    frequencies = np.array(frequencies)[indices]
    max_freq_index = frequencies < max_freq

    
    RVEL_theta = np.interp(freqs, frequencies[max_freq_index], RVEL_theta[max_freq_index])
    RVEL_phi = np.interp(freqs, frequencies[max_freq_index], RVEL_phi[max_freq_index])


    antenna_response["theta"] = RVEL_theta
    antenna_response["phi"] = RVEL_phi
    return antenna_response



class channelThermalNoiseAdder:
    """
    module to generate thermal noise, both from electronics and ice
    this class is a stripped version of channelGalacticNoiseAdder, modified to
    generate thermal noise instead of galactic radio noise
    """

    def __init__(self):
        self.__n_side = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    

    @functools.lru_cache(maxsize=1024 * 24 * 32)
    def get_cached_antenna_response(self, antenna_pattern, zen, azi, *ant_orient, use_db=True):
        if isinstance(antenna_pattern, tuple):
            return get_antenna_response_from_pickle_content(antenna_pattern, self.freqs, zen, azi)
        else:
            return antenna_pattern.get_antenna_response_vectorized(self.freqs, zen, azi, *ant_orient)


    def get_temperature_at_depth(self, depth):
        grip_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../../NuRadioMC/data"
        grip_file = "griptemp.txt"
        grip_path = grip_dir + "/" + grip_file

        if not os.path.exists(grip_path):
            grip_url = "https://doi.pangaea.de/10.1594/PANGAEA.89007?format=textfile"
            urlretrieve(grip_url, grip_path)

        grip_temp = np.loadtxt(grip_path, skiprows=39)
        depth_values = -grip_temp[:,0]
        temperature_values = grip_temp[:,1]
        profile = interp1d(depth_values,
                           temperature_values,
                           bounds_error=False,
                           fill_value=(temperature_values[-1],temperature_values[0]))
        return profile(depth)

    def solid_angle(self, theta, d_theta, d_phi):
        return np.abs(np.sin(theta) * np.sin(d_theta / 2) * 2 * d_phi)

    def get_temperature_from_json(self, temperature_file):
        with open(temperature_file, "r") as file_open:
            temperature_file_dict = json.load(file_open)
        z_antenna = temperature_file_dict["z_antenna"]
        theta = temperature_file_dict["theta"] * units.rad
        eff_temperature = temperature_file_dict["eff_temperature"]
        return z_antenna, theta, eff_temperature



    def begin(self, sim_library_dir, nr_phi_bins=64, debug=False):
        """
        Set up important parameters for the module

        Parameters
        ----------
        n_side: int, default: 4
            The n_side parameter of the healpix map. Has to be power of 2
            The skymap is downsized to the resolution specified by the n_side
            parameter and for every pixel above the horizon the antenna's vector effective length
            from that direction is folded into the generated thermal noise.
            The number of pixels used is 12 * n_side ** 2, so a larger value for n_side will result better accuracy
            but also greatly increase computing time.
        noise_temperature: float, default: 300
            The noise temperature of the ambient ice in Kelvin.
        """
        self.temperature_files = [f"{sim_library_dir}/eff_temperature_-100m_ntheta100_GL3.json",
                                  f"{sim_library_dir}/eff_temperature_-40m_ntheta100.json",
                                  f"{sim_library_dir}/eff_temperature_-1.0m_ntheta100_GL3.json",
                                  f"{sim_library_dir}/eff_temperature_-2.0m_ntheta100_GL3.json", 
                                  f"{sim_library_dir}/eff_temperature_-3.0m_ntheta100_GL3.json", 
                                  ]

        self.eff_temperature = {}
        for temperature_file in self.temperature_files:
            z_antenna, self.thetas, eff_temperature = self.get_temperature_from_json(temperature_file)
            self.eff_temperature[z_antenna] = eff_temperature

        self.nr_theta_bins = len(self.thetas)
#        self.channel_depths = {0 : -100,
#                               4 : -100,
#                               7 : -40, 12: -1.0, 13: -1.0}
        self.channel_depths = {i : -100 for i in [0, 1, 2, 3, 4, 8, 9, 10, 11, 21, 22, 23]}
        for i in [5, 6, 7]:
            self.channel_depths[i] = -40
        for i in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
            self.channel_depths[i] = -1.0

        self.debug = debug
        if self.debug:
            nr_phi_bins = 16
        self.phis = np.linspace(0 * units.degree, 360 * units.degree, nr_phi_bins)
        return


    @register_run()
    def run(
            self,
            event,
            station,
            detector,
            passband=None
    ):

        """
        Adds noise resulting from thermal emission to the channel traces

        Parameters
        ----------
        event: Event object
            The event containing the station to whose channels noise shall be added
        station: Station object
            The station whose channels noise shall be added to
        detector: Detector object
            The detector description
        passband: list of float, optional
            Lower and upper bound of the frequency range in which noise shall be
            added. The default (no passband specified) is [10, 1600] MHz
        """

        # check that for all channels channel.get_frequencies() is identical
        last_freqs = None
        for channel in station.iter_channels():
            if last_freqs is not None and (
                    not np.allclose(last_freqs, channel.get_frequencies(), rtol=0, atol=0.1 * units.MHz)):
                logger.error("The frequencies of each channel must be the same, but they are not!")
                return

            last_freqs = channel.get_frequencies()

        freqs = last_freqs
        self.freqs = freqs
        d_f = freqs[2] - freqs[1]

        if passband is None:
            passband = [10 * units.MHz, 1600 * units.MHz]

        passband_filter = (freqs > passband[0]) & (freqs < passband[1])

        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        station_time = station.get_station_time()


        n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
        n_air = ice.get_refractive_index(depth=1, site=detector.get_site(station.get_id()))
        c_vac = scipy.constants.c * units.m / units.s

        channel_spectra = {}
        for channel in station.iter_channels():
            channel_spectra[channel.get_id()] = channel.get_frequency_spectrum()


        # need exception for these antenna models since the queried RVELs from NuRadio are not correct,
        # only the pickle files contain correct profiles
        # if the pickle files are not on your system they are automatically downloaded
        magphase_antenna_patterns = [f"vpol_effective_height_n1pt{n}_5x_upsample" for n in [3, 4, 5, 6, 7]] 
        antenna_model = detector.get_antenna_model(station.get_id(), channel.get_id())
        if antenna_model in magphase_antenna_patterns:
            antenna_dir = "/user/rcamphyn/software/NuRadioMC/NuRadioReco/detector/AntennaModels/"
#            antenna_path = antenna_dir + antenna_model + "/" + antenna_model + ".pkl"
#            antenna_pattern = NuRadioReco.detector.antennapattern.get_pickle_antenna_response(antenna_path)
#            # to make hashable for lru_cache
#            antenna_pattern = structure_pickle_antenna_pattern(antenna_pattern)
#            antenna_pattern = make_hashable(antenna_pattern)
            antenna_pattern = NuRadioReco.detector.antennapattern.AntennaPattern(antenna_model, path=antenna_dir)
        else:
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(
                antenna_model,
    #                        interpolation_method="magphase"
                )
        antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel.get_id())

        d_thetas = np.diff(self.thetas)
        d_phis = np.diff(self.phis)
#        d_phi = 2 * np.pi
        channel_ids = station.get_channel_ids()
        for phi, d_phi in zip(self.phis, d_phis):
            for theta_i, (theta, d_theta) in enumerate(zip(self.thetas, d_thetas)):
                solid_angle = self.solid_angle(theta, d_theta, d_phi)

                noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=complex)
                channel_noise_spec = np.zeros_like(noise_spectrum)
                for channel_id in channel_ids:
                    depth = self.channel_depths[channel_id]
                    eff_temperature = self.eff_temperature[depth]
                    # calculate spectral radiance of radio signal using rayleigh-jeans law
                    spectral_radiance = (2. * (scipy.constants.Boltzmann * units.joule / units.kelvin)
                        * freqs[passband_filter] ** 2 * eff_temperature[theta_i] * solid_angle / c_vac ** 2)
                    spectral_radiance[np.isnan(spectral_radiance)] = 0

                    # calculate radiance per energy bin
                    spectral_radiance_per_bin = spectral_radiance * d_f

                    # calculate electric field per frequency bin from the radiance per bin
                    efield_amplitude = np.sqrt(
                        spectral_radiance_per_bin / (c_vac * scipy.constants.epsilon_0 * (
                                units.coulomb / units.V / units.m))) / d_f

                    # assign random phases to electric field
                    if self.debug:
                        phases = 0 * np.ones(len(spectral_radiance))
                    else:
                        phases = np.random.uniform(0, 2. * np.pi, len(spectral_radiance))

                    noise_spectrum[1][passband_filter] = np.exp(1j * phases) * efield_amplitude
                    noise_spectrum[2][passband_filter] = np.exp(1j * phases) * efield_amplitude



                    # add random polarizations and phase to electric field
                    if self.debug:
                        polarizations = [0.] * len(spectral_radiance)
                    else:
                        polarizations = np.random.uniform(0, 2. * np.pi, len(spectral_radiance))

                    channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.cos(polarizations)
                    channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.sin(polarizations)

                    # fold electric field with antenna response
    #                antenna_response = antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth,
    #                                                                                   *antenna_orientation)

                    antenna_response = self.get_cached_antenna_response(antenna_pattern, theta, phi,
                                                                        *antenna_orientation)

                    channel_noise_spectrum = (
                        antenna_response['theta'] * channel_noise_spec[1]
                        + antenna_response['phi'] * channel_noise_spec[2]
                    )

                    # add noise spectrum from pixel in the sky to channel spectrum
                    channel_spectra[channel_id] += channel_noise_spectrum

        # store the updated channel spectra
        for channel in station.iter_channels():
            channel.set_frequency_spectrum(channel_spectra[channel.get_id()], "same")


if __name__ == "__main__":
    import datetime
    import matplotlib.pyplot as plt
    from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
    from NuRadioReco.framework.event import Event
    from NuRadioReco.framework.station import Station
    from NuRadioReco.framework.channel import Channel

    # SETTINGS
    station_id = 11
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    channel_ids = [0]
#    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_IGLU_n1.74",
#                      "HPol" : "RNOG_hpol_v4_8inch_center_IGLU_n1.74"}
    antenna_models = {"VPol" : "vpol_effective_height_n1pt3_5x_upsample",
                      "HPol" : "RNOG_hpol_v4_8inch_center_IGLU_n1.74"}
    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
                     "HPol" : [4, 8, 11, 21]}
    detector_time = datetime.datetime(2023, 8, 1)



    detector = ModDetector(database_connection='RNOG_public', log_level=logging.NOTSET,
                           select_stations=station_id)
    detector.update(detector_time)

    for channel_id in channel_ids:
        if channel_id in channel_types["VPol"]:
            antenna_model = antenna_models["VPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain", "VEL"], antenna_model)
        if channel_id in channel_types["HPol"]:
            antenna_model = antenna_models["HPol"]
            detector.modify_channel_description(station_id, channel_id, ["signal_chain", "VEL"], antenna_model)


    t0 = datetime.datetime.now()
    spectra = []
    nr_it = 100
    thermal_noise_adder = channelThermalNoiseAdder()
    thermal_noise_adder.begin(sim_library_dir="sim/library", debug=False)
    for i in range(nr_it):
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)

        thermal_noise_adder.run(event, station, detector)

        station = event.get_station()
        channel = station.get_channel(0)
        spectra.append(np.abs(channel.get_frequency_spectrum()))
    t1 = datetime.datetime.now()
    print(f"took {t1 - t0} to run {nr_it} iterations")


    plt.plot(channel.get_frequencies(), np.mean(spectra, axis=0))
    plt.show()
    plt.savefig("test_thermal_noise")
