import numpy as np
import pickle
from scipy import constants

import matplotlib.pyplot as plt

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import units


def temp_to_volt(temp, bandwidth):
    """
    bandwidth: int
        frequency bandwidth of simulated noise
        expected to be in GHz

    returns
    -------
    voltage: float
        voltage in NuRadio base units
    """
    # check where this value comes from, it's often used in RNO-G for this formula to describe electronic noise
    resistance = 50 * units.ohm
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    voltage = np.sqrt(4 * k * temp * resistance * bandwidth)
    return voltage

class spectralAmplitudeHistogramSimulator():
    """
    Class formalism to yield simulated spectral amlitude histograms
    A spectral amplitude histogram is made by taking the spectral amplitude at a chosen frequency
    for a chosen amount of spectra and plotting these in a histogram. In such a way, one histogram is made for
    every sampled frequeny, yielding nr_samples/2 + 1 histograms
    (/2 due to the Nyquist theorem and + 1 for freq = 0 Hz but this can mostly be negated since the DC component is often already removed)
    """

    def __init__(self):
        pass

    def begin(self, station_id, detector, temp, bandwidth, filter_bandpass):
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

        self.bandwidth = bandwidth 
        self.amplitude = temp_to_volt(self.temp, self.bandwidth)

        self.channel_bandpass_filter = channelBandPassFilter()
        self.filter_bandpass = filter_bandpass


    def simulate(self, sim_samples, nr_bins, hist_range, normalized=True, savefile=None):
        """
        sim_samples: int
            nr of frequency spectra to be simulated
        """
        bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)

        channelgenericnoiseadder = channelGenericNoiseAdder()

        spec_amplitude_histograms = np.zeros((len(self.channel_ids), len(self.frequencies), nr_bins))

 

        for _ in range(sim_samples):
            noise_spectrum = channelgenericnoiseadder.bandlimited_noise(
                    None,
                    None,
                    self.nr_samples, self.sampling_rate, self.amplitude,
                    type="rayleigh", time_domain=False)

            for channel_id in self.channel_ids:
                det_resp = self.detector.get_signal_chain_response(self.station_id, channel_id)
                det_resp = det_resp(np.array(self.frequencies))

                if self.filter_bandpass:
                    bandpassfilter = self.channel_bandpass_filter.get_filter(self.frequencies, 
                                                                             self.station_id, channel_id, self.detector,
                                                                             passband=self.filter_bandpass, filter_type="rectangular")

                    det_resp = det_resp * bandpassfilter
                
                noise_spectrum_with_detector = noise_spectrum * det_resp

                # bin edges in searchsorted should only be inner edges
                bin_indices = np.searchsorted(bin_edges[1:-1],
                                              np.abs(noise_spectrum_with_detector))
                print(bin_indices.shape)
                
                spec_amplitude_histograms[channel_id, np.arange(len(self.frequencies)), bin_indices] += 1


        if normalized:
            normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
            spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, :, np.newaxis])

        if savefile:
            with open(savefile, "wb") as file:
                # unphysical number of runs to indicate this is simulation
                nr_runs = -1
                pickle_dir = {"freqs": self.frequencies, "spec_hists": spec_amplitude_histograms, "sampling_rate": self.sampling_rate, "nr_runs": nr_runs}
                pickle.dump(pickle_dir, file)
        
        return self.frequencies, spec_amplitude_histograms


    def simulate_single_frequency(self, frequency, sim_samples, nr_bins, hist_range, normalized=True, savefile=None):

        """
        purely for testing purposes

        frequency : float
            frequency in GHz
        sim_samples: int
            nr of frequency spectra to be simulated
        """
        frequency = min(self.frequencies, key=lambda f:abs(f-frequency))
        bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins + 1)

        channelgenericnoiseadder = channelGenericNoiseAdder()

        spec_amplitude_histograms = np.zeros((len(self.channel_ids), nr_bins))

        for _ in range(sim_samples):
            noise_spectrum = channelgenericnoiseadder.bandlimited_noise(
                    None,
                    None,
                    self.nr_samples, self.sampling_rate, self.amplitude,
                    type="rayleigh", time_domain=False)



            for channel_id in self.channel_ids:
                det_resp = self.detector.get_signal_chain_response(self.station_id, channel_id)
            
                det_resp = det_resp(np.array([frequency]))

                noise_spectrum_with_detector = noise_spectrum * det_resp
                noise_spectrum_at_freq = noise_spectrum_with_detector[np.where(frequency == self.frequencies)]
                noise_spectrum_at_freq = np.abs(noise_spectrum_at_freq)


                # bin edges in searchsorted should only be inner edges
                bin_idx = np.searchsorted(bin_edges[1:-1],
                                          noise_spectrum_at_freq)
                
                spec_amplitude_histograms[channel_id, bin_idx] += 1


        if normalized:
            normalization_factor = np.sum(spec_amplitude_histograms, axis = -1) * np.diff(bin_edges)[0]
            spec_amplitude_histograms = np.divide(spec_amplitude_histograms, normalization_factor[:, np.newaxis])

        if savefile:
            with open(savefile, "wb") as file:
                # unphysical number of runs to indicate this is simulation
                nr_runs = -1
                pickle_dir = {"freqs": self.frequencies, "spec_hists": spec_amplitude_histograms, "sampling_rate": self.sampling_rate, "nr_runs": nr_runs}
                pickle.dump(pickle_dir, file)

        return frequency, spec_amplitude_histograms
