import numpy as np
import pickle
from scipy import constants

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
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
        voltage in milivolts
    """
    # check where this value comes from, it's often used in RNO-G for this formula to describe electronic noise
    resistance = 50 * units.ohm
    k = constants.k * (units.m**2 * units.kg * units.second**-2 * units.kelvin**-1)
    voltage = np.sqrt(4 * k * temp * resistance * bandwidth)
    return voltage / units.mV

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

    def begin(self, station, detector, temp, bandwidth):
        """
        temp : int
            temperature in kelvin
        bandpass : list
            frequencies in GHz
        """
        self.station = station
        self.detector = detector
        self.channels = detector.get_channel_ids(station)
        self.channels = sorted(self.channels)
        self.temp = temp

        self.bandwidth = bandwidth 
        self.amplitude = temp_to_volt(self.temp, self.bandwidth)


    def simulate(self, sim_samples, nr_bins, hist_range, normalized=True, savefile=None):
        """
        sim_samples: int
            nr of frequency spectra to be simulated
        """
        bin_edges = np.linspace(hist_range[0], hist_range[1], nr_bins+1)

        channelgenericnoiseadder = channelGenericNoiseAdder()

        frequencies = []
        spec_hists = []
        print(self.channels)
        for channel in self.channels:
            print(channel)
            sampling_rate = self.detector.get_sampling_frequency(self.station, channel)
            nr_samples = self.detector.get_number_of_samples(self.station, channel)
            det_resp = self.detector.get_signal_chain_response(self.station, channel)
            
            # freqs are in GHz since this is base unit of NuRadio
            channel_frequencies = np.fft.rfftfreq(nr_samples, 1./sampling_rate)
            det_resp = np.array([det_resp(f) for f in channel_frequencies])
            
            channel_spec_hists = np.zeros((len(channel_frequencies), nr_bins))
            for _ in range(sim_samples):
                noise_spectrum = channelgenericnoiseadder.bandlimited_noise(None, None, nr_samples, sampling_rate, self.amplitude,
                                                                            type="rayleigh", time_domain=False)
                filt_spectrum = noise_spectrum * det_resp

                # bin edges in digitize only take inner edges
                bin_idxs = np.digitize(np.abs(noise_spectrum), bin_edges[1:-1])
                for i in range(len(channel_frequencies)):
                    channel_spec_hists[i, bin_idxs[i]] += 1

            frequencies.append(channel_frequencies)
            spec_hists.append(channel_spec_hists)

        if normalized:
            normalization_factor = np.sum(spec_hists, axis = -1) * np.diff(bin_edges)[0]
            spec_hists = np.divide(spec_hists, normalization_factor[:, :, np.newaxis])

        if savefile:
            with open(savefile, "wb") as file:
                # unphysical to indicate this is simulation
                nr_runs = -1
                pickle_dir = {"freqs": frequencies, "spec_hists": spec_hists, "sampling_rate": sampling_rate, "nr_runs": nr_runs}
                pickle.dump(pickle_dir, file)
        
        return frequencies, spec_hists
