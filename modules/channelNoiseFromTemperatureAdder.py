import matplotlib.pyplot as plt
from numpy.random import Generator, Philox
import numpy as np
import pandas as pd


from NuRadioReco.utilities import constants, fft, units
from NuRadioReco.utilities.signal_processing import calculate_vrms_from_temperature






class channelNoiseFromTemperatureAdder():
        
    # taken from NuRadio's channelGenericNoiseAdder
    def add_random_phases(self, amps, n_samples_time_domain):
        """
        Adding random phase information to given amplitude spectrum.

        Parameters
        ----------

        amps: array of floats
            Data that random phase is added to.
        n_samples_time_domain: int
            number of samples in the time domain to differentiate between odd and even number of samples
        """
        amps = np.array(amps, dtype='complex')
        Np = (n_samples_time_domain - 1) // 2
        phases = self.__random_generator.random(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        amps[1:Np + 1] *= phases  # Note that the last entry of the index slice is f[Np] !

        return amps


    def __init__(self, seed=None):
        self.__random_generator = Generator(Philox(seed))
        return

    def begin(self, noise_temperature_path : str):
        """
        path is a csv file
        len 2 describes downhole and surface
        """
        ds_downhole = pd.read_csv(noise_temperature_path[0], names=["freq", "noise_temp"], header=None, index_col=False, skipinitialspace=True)
        ds_surface = pd.read_csv(noise_temperature_path[1], names=["freq", "noise_temp"], header=None, index_col=False, skipinitialspace=True)


        # in this range the temperatures make sense, seen from the plots and set by Nat and Eric
        clean_frequency_range = [0.1, 0.7]

        self.frequencies = ds_downhole["freq"] * units.MHz
        assert np.all(self.frequencies == ds_surface["freq"] * units.MHz)

        unclean_frequency_selection_left = (self.frequencies <= clean_frequency_range[0])
        unclean_frequency_selection_right = (self.frequencies >= clean_frequency_range[1])

        self.noise_temperatures = {}
        self.noise_temperatures["deep"] = ds_downhole["noise_temp"].to_numpy() * units.kelvin
        self.noise_temperatures["surface"] = ds_surface["noise_temp"].to_numpy() * units.kelvin


        for key in ["deep", "surface"]:
            self.noise_temperatures[key][unclean_frequency_selection_left] = self.noise_temperatures[key][np.invert(unclean_frequency_selection_left)][0] 
            self.noise_temperatures[key][unclean_frequency_selection_right] = self.noise_temperatures[key][np.invert(unclean_frequency_selection_right)][-1] 




        self.selector = {ch_id : "deep" for ch_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 21, 22, 23]}
        for ch_id in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
            self.selector[ch_id] = "surface"


    def get_spectrum(self, station_id, channel_id, sampling_rate, nr_samples):
        noise_temperatures = self.noise_temperatures[self.selector[channel_id]]
        spectrum = self.convert_temp_to_spectrum(self.frequencies, noise_temperatures, sampling_rate, nr_samples)
        return spectrum



    def convert_temp_to_spectrum(self, frequencies, temperatures, sampling_rate, nr_samples, impedance=50*units.ohm, mode="rayleigh"):
        final_frequencies = fft.freqs(nr_samples, sampling_rate)
        df = np.diff(frequencies)[0]
        frequency_bins_lower = frequencies - df/2.
        frequency_bins_upper = frequencies + df/2.
        nrbinsactive = []
        for i, _ in enumerate(frequency_bins_lower):
            binsactive_tmp = (final_frequencies >= frequency_bins_lower[i]) & (final_frequencies <= frequency_bins_upper[i])
            nrbinsactive.append(np.sum(binsactive_tmp))

        vrms_per_bin = np.sqrt(constants.k_B * temperatures * df * impedance)
        amplitudes = 1. * nr_samples / np.sqrt(nrbinsactive) * vrms_per_bin
        spectrum = np.zeros_like(final_frequencies)
        if mode == "rayleigh":
            amplitudes /= np.sqrt(2)
            for bin_nr, _ in enumerate(frequency_bins_lower):
                selection = (final_frequencies >= frequency_bins_lower[bin_nr]) & (final_frequencies <= frequency_bins_upper[bin_nr])
                spectrum[selection] = self.__random_generator.rayleigh(amplitudes[bin_nr], np.sum(selection))
        spectrum = self.add_random_phases(spectrum, nr_samples) / sampling_rate
        return spectrum



    def run(self, event, station, detector):
        station_id = station.get_id()
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            sampling_rate = detector.get_sampling_frequency(station_id, channel_id)
            nr_samples = detector.get_number_of_samples(station_id, channel_id)
            channel_trace = channel.get_trace()
            noise_spectrum = self.get_spectrum(station_id, channel_id, sampling_rate, nr_samples)
            noise_trace = fft.freq2time(noise_spectrum, sampling_rate)
            channel.set_trace(channel_trace + noise_trace, sampling_rate)
        








if __name__ == "__main__":
    from astropy.time import Time
    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    
    class Station():
        def __init__(self, id):
            self.id = id
            return
        def get_id(self):
            return self.id


    station_id = 11
    station = Station(station_id)
    det = Detector(select_stations=station_id,
                   signal_chain_measurement_name="calibrated_impulse_response_v0")
    det_time = Time("2023-08-01")
    det.update(det_time)


    test_paths = ["/user/rcamphyn/noise_study/RF_chain_testing/october_drabsurf_freezer/Calibrated_noise_figures/Deep_calibrated_noisetemp_-40C.csv",
                "/user/rcamphyn/noise_study/RF_chain_testing/october_drabsurf_freezer/Calibrated_noise_figures/Surface_calibrated_noisetemp_20C.csv"]
    noise_adder = channelNoiseFromTemperatureAdder()
    noise_adder.begin(test_paths)
    noise_adder.run(event=0, station=station, detector=det)
