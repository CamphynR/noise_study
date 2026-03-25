import copy
from collections.abc import Iterable
from iminuit import Minuit 
from iminuit.cost import LeastSquares
from iminuit import util
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from tabulate import tabulate

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.utilities import units
from NuRadioReco.utilities.fft import freq2time, time2freq

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator
from modules.cableResponse import cableResponse
from utilities.utility_functions import read_pickle, find_config, read_config

def find_frequency_peaks(freq: np.ndarray, spectrum : np.ndarray, threshold : float = 4):
    
    rms = np.sqrt(np.mean(np.abs(spectrum)**2))
    peak_idxs = np.where(np.abs(spectrum) > threshold * rms)[0]

    return peak_idxs

def calculate_reduced_chi2(spectrum_1, spectrum_2, frequencies, freq_range):
    mask = (frequencies > freq_range[0]) & (frequencies < freq_range[1])
    spectrum_1 = np.abs(spectrum_1[mask])
    spectrum_2 = np.abs(spectrum_2[mask])
    ndof = len(spectrum_1)

    chi2 = np.sum((spectrum_1 - spectrum_2)**2)
    chi2_reduced = chi2/ndof

    return chi2_reduced



def read_freq_spec_file(path):
    """
    Reads the pickle file that contains frequency and spectra data
    """
    result_dictionary = copy.deepcopy(read_pickle(path))
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    header = result_dictionary["header"]
    return frequencies, frequency_spectrum, var_frequency_spectrum, header



class spectrumFitter:
    def __init__(self, data_path, sim_paths,
                 cross_products_path = None,
                 sampling_rate=3.2 * units.GHz,
                 fit_range=[0.25, 0.6],
                 bandpass=None, bandpass_type="butter",
                 system_response=None,
                 cost_function = None,
                 fit_function_parameter_guesses=None,
                 goodness_of_fit_function=calculate_reduced_chi2,
                 remove_cable=True,
                 cable_length=11):
        """
        sim_paths : list
            list of all simulation components, without the sum
            the order is assumed to be ice, electronic, galactic, rest
        system_response : array or systemResponseTimeDomainIncorporator instance
            the module can accept a system response to apply to the simulations
        """

        config_path = find_config(sim_paths[0], sim=True)
        self.config = read_config(config_path)


        vpols = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
        hpols = [4, 8, 11, 21]
        lpdas = [12, 13, 14, 15, 16, 17, 18, 19, 20]    


        self.channels_to_include = self.config["channels_to_include"]

        self.sampling_rate = sampling_rate
        self.frequencies, \
        self.data_spectrum, \
        self.var_data_spectrum, \
        self.data_header = read_freq_spec_file(data_path)

        self.goodness_of_fit_function = goodness_of_fit_function

        if isinstance(system_response, systemResponseTimeDomainIncorporator):
            system_response = np.array([system_response.get_response(c)["gain"](self.frequencies) for c in range(len(self.channels_to_include))])
            for gain in system_response:
                assert np.max(gain) == 1.
        


        # 11m CABLE response that is between LPDA and SURFACE
        # This cable was used in measuring the electronic noise for the surface components
        # so has to be divided out of the eletronic noise to avoid double counting
        # with the cable in the system response template

        cable_response_helper = cableResponse(length=cable_length)
        self.cable_response = cable_response_helper.get_gain()(self.frequencies)



        self.sim_spectra = []
        self.sim_var_spectra = []
        self.sim_headers = []
        for i, sim_path in enumerate(sim_paths):
            sim_frequencies_i, \
            sim_spectrum_i, \
            sim_var_spectrum_i, \
            sim_header = read_freq_spec_file(sim_path)
            assert np.all(self.frequencies == sim_frequencies_i)
            if system_response is not None:
                sim_spectrum_i = sim_spectrum_i * system_response
                sim_var_spectrum_i = sim_var_spectrum_i * system_response**2
                # divide out cable from electronic noise in lpdas
                if i == 1 and remove_cable:
                    for channel_id in lpdas:
                        sim_spectrum_i[channel_id] = sim_spectrum_i[channel_id] / self.cable_response / self.cable_response
            self.sim_spectra.append(sim_spectrum_i)
            self.sim_var_spectra.append(sim_var_spectrum_i)
            self.sim_headers.append(sim_header)

        if cross_products_path:
            cross_products = read_pickle(cross_products_path)
            self.cross_products = [
                    cross_products["ice_el_cross"],
                    cross_products["ice_gal_cross"],
                    cross_products["el_gal_cross"]
                    ]

            # save as [ice_el, ice_gal, el_gal]
            for cross_i, cross_product in enumerate(self.cross_products):
                if system_response is not None:
                    self.cross_products[cross_i] = system_response * cross_product
                # divide out cable from electronic noise in lpdas
                if cross_i in [0, 2] and remove_cable:
                    for channel_id in lpdas:
                        self.cross_products[cross_i][channel_id] = self.cross_products[cross_i][channel_id] / self.cable_response

            
            for channel_id in self.channels_to_include: 
                ice_el_cross = self.cross_products[0][channel_id]
                ice_gal_cross = self.cross_products[1][channel_id]
                el_gal_cross = self.cross_products[2][channel_id]
                assert np.all(np.imag(ice_el_cross + ice_gal_cross + el_gal_cross) < 1e-20), "imaginary part of cross product sum should be 0"

                



        self.fit_range = fit_range
        fit_band = np.where(np.logical_and(fit_range[0] < self.frequencies,
                                           self.frequencies < fit_range[1]))
#        self.fit_idxs = {}
        # we filter out any CW's from the fit range out of upward facing lpdas
        # (peak finding algorihtm should not be used for other antenna types)
#        lpdas_up = [13, 16, 19]
#        for channel_id in self.channels_to_include:
#            if channel_id in lpdas_up:
#                peaks_freq_idxs = find_frequency_peaks(self.frequencies,
#                                                       self.data_spectrum[channel_id],
#                                                       threshold=2)
#                peak_width_idxs = int(10 * units.MHz/np.diff(self.frequencies[:2]))
#                idxs_to_remove = [np.arange(peak_freq_idx - peak_width_idxs/2, peak_freq_idx + peak_width_idxs/2)
#                                                for peak_freq_idx in peaks_freq_idxs]
#
#                freqs_no_cw = np.delete(self.frequencies,idxs_to_remove)
#
#                import matplotlib.pyplot as plt
#                plt.style.use("retro")
#                plt.plot(frequencies, self.data_spectrum[channel_id])
#                plt.scatter(freqs_no_cw, np.median(self.data_spectrum[channel_id]))
#                plt.xlim(0., 1.)
#                plt.savefig("test")
#                exit()
#            else:
#                self.fit_idxs[channel_id] = fit_band


        self.fit_idxs = np.where(np.logical_and(fit_range[0] < self.frequencies,
                                                self.frequencies < fit_range[1]))

        self.bandpass = bandpass
        if self.bandpass is None:
            self.bandpass = [0.1, 0.7]

        self.bandpass_type = bandpass_type

        self.fit_function_parameter_guesses = fit_function_parameter_guesses

        if cost_function is None:
            self.cost_function = LeastSquares
        else:
            self.cost_function = cost_function




        return

    def get_fit_gain(self, mode="constant", choose_channels=None, parameter_limits=None):
        if choose_channels is None:
            channel_ids = self.channels_to_include
        else:
            channel_ids = choose_channels


        fit_results = []
        goodness_of_fits = []
        for channel_id in channel_ids:
            x_data = self.frequencies[self.fit_idxs]
            y_data = self.data_spectrum[channel_id][self.fit_idxs]
            y_err = self.var_data_spectrum[channel_id][self.fit_idxs]

            if mode == "constant":
                fit_function = self.select_fit_function(mode, channel_id)
                cost_function = self.cost_function(x=x_data, y=y_data,
                                             yerror=y_err,
                                             model=fit_function)
                m = Minuit(cost_function, gain=1000.)

                if parameter_limits is None:
                    parameter_limits = [(0, 3000)]
                m.limits = parameter_limits

                m.migrad()
                m.hesse()
                fit_results.append(m.params)
            elif mode == "electronic_temp":
                fit_function = self.select_fit_function(mode, channel_id)
                cost_function = self.cost_function(x=x_data, y=y_data,
                                             yerror=y_err,
                                             model=fit_function)
                if self.fit_function_parameter_guesses is None:
                    self.fit_function_parameter_guesses = dict(
                            gain=1000.,
                            el_ampl=0.,
                            el_cst=1.,
                            f0=0.15
                            )
                m = Minuit(cost_function, **self.fit_function_parameter_guesses)
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True

                if parameter_limits is None:
                    parameter_limits = [(0., 1500), (0., None), (0., None), (0., 0.6)]
                m.limits = parameter_limits

                m.migrad()
                m.fixed["gain"] = True
                m.fixed["el_ampl"] = False
                m.fixed["el_cst"] = False
                m.fixed["f0"] = True
                m.migrad()
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True
                m.migrad()
#                m.fixed["gain"] = False
#                m.fixed["el_ampl"] = False
#                m.fixed["el_cst"] = False
#                m.fixed["f0"] = True
#                m.migrad()
                m.hesse()
                fit_results.append(m.params)
            elif mode == "electronic_temp_cross":
                fit_function = self.select_fit_function(mode, channel_id)
                cost_function = self.cost_function(x=x_data, y=y_data,
                                             yerror=y_err,
                                             model=fit_function)
                if self.fit_function_parameter_guesses is None:
                    self.fit_function_parameter_guesses = dict(
                            gain=1000.,
                            el_ampl=0.,
                            el_cst=1.,
                            f0=0.15
                            )
                m = Minuit(cost_function, **self.fit_function_parameter_guesses)
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True

                if parameter_limits is None:
                    parameter_limits = [(200., 3000), (-10, 10.), (0., None), (0., 0.7)]

                m.limits = parameter_limits
                m.migrad()
                m.fixed["gain"] = True
                m.fixed["el_ampl"] = False
                m.fixed["el_cst"] = True
                m.fixed["f0"] = False
                m.migrad()
                m.fixed["gain"] = False
                m.fixed["el_ampl"] = True
                m.fixed["el_cst"] = True
                m.fixed["f0"] = True
                m.migrad()
#                m.fixed["gain"] = False
#                m.fixed["el_ampl"] = False
#                m.fixed["el_cst"] = False
#                m.fixed["f0"] = True
#                m.migrad()
                m.hesse()
                fit_results.append(m.params)
            elif isinstance(mode, dict):
                if self.fit_function_parameter_guesses is None:
                    self.fit_function_parameter_guesses = dict(
                            gain=1000.,
                            el_ampl=0.,
                            el_cst=1.,
                            f0=0.15
                            )
                fit_function = self.select_fit_function(mode["fit_function"], channel_id)
                cost_function = self.cost_function(x=x_data, y=y_data,
                                             yerror=y_err,
                                             model=fit_function)
                m = Minuit(cost_function, **self.fit_function_parameter_guesses)
                m.limits = [(0., 1500), (0., None), (0., None), (0., 0.6)]
                for step_nr, step in mode["steps"].items():
                    m.fixed["gain"] = step["gain"]
                    m.fixed["el_ampl"] = step["el_ampl"]
                    m.fixed["el_cst"] = step["el_cst"]
                    m.fixed["f0"] = step["f0"]
                    m.migrad()
                    m.hesse()
                    print(m.covariance)
                fit_results.append(m.params)

            # calculate goodness of fit

            fit_param_list = [param.value for param in m.params]
            sim_spectrum = fit_function(x_data, *fit_param_list)
            gof = self.goodness_of_fit_function(y_data, np.abs(sim_spectrum),
                                                x_data, freq_range=self.fit_range)
            goodness_of_fits.append(gof)


        return fit_results, goodness_of_fits



    def save_fit_results(self, mode="electronic_temp", parameter_limits=None,
                         save_folder=None, filename=None, extended=True):
        """
        If extended is True, save all fit parameters,
        otherwise only save gain

        Also returns the fit results to use in plotting
        """
        if save_folder is None:
            save_folder = os.path.dirname(__file__)

        fit_results, goodness_of_fits = self.get_fit_gain(mode=mode, parameter_limits=parameter_limits)

        if extended:
            value_dicts = [{f.name : f.value for f in fit_result} for fit_result in fit_results]
            error_dicts = [{f.name : f.error for f in fit_result} for fit_result in fit_results]
        else:
            value_dicts = [{"gain" : fit_result["gain"].value} for fit_result in fit_results]
            error_dicts = [{"gain" : fit_result["gain"].error} for fit_result in fit_results]

        value_df = pd.DataFrame(value_dicts)
        error_df = pd.DataFrame(error_dicts)

        header=True if extended else False
        if filename is None:
            season = self.data_header["begin_time"].strftime("%Y")[0]
            filename = f"absolute_amplitude_calibration_season{season}_st{self.config['station']}.csv"
            filename_error = f"absolute_amplitude_calibration_season{season}_st{self.config['station']}_errors.csv"
        else:
            filename_error = filename.split(".csv")[0] + "error" + ".csv"
        value_df.to_csv(os.path.join(save_folder, filename), header=header)
        error_df.to_csv(os.path.join(save_folder, filename_error), header=header)

        return fit_results, goodness_of_fits
        



#        fieldnames = self.channels_to_include
#        with open(filename, "w") as file:
#            writer = csv.DictWriter(file, fieldnames=fieldnames)
#            writer.writeheader()
#            writer.writerow({ch: fit_results[ch]["gain"] for ch in fieldnames})
#            if extended:
#                writer.writerow({ch: fit_results[ch]})
        

        return


    def get_fit_range(self):
        return self.fit_range

    def set_fit_range(self, fit_range):
        self.fit_range = fit_range
        return


    
    def get_fit_function(self, mode, channel_id):
        fit_function = self.select_fit_function(mode, channel_id)
        return fit_function



    def select_fit_function(self, mode, channel_id):
        if mode == "constant":
            ice_spectrum = self.sim_spectra[0][channel_id]
            electronic_spectrum = self.sim_spectra[1][channel_id]
            galactic_spectrum = self.sim_spectra[2][channel_id]

            ice_el_cross = self.cross_products[0][channel_id]
            ice_gal_cross = self.cross_products[1][channel_id]
            el_gal_cross = self.cross_products[2][channel_id]

            def fit_gain_factor(freq, gain):

                combined_spectrum = \
                        gain * np.sqrt( \
                        ice_spectrum**2 + electronic_spectrum**2 + galactic_spectrum**2 
                    +   ice_el_cross + ice_gal_cross + el_gal_cross ) 

                function_interp = interp1d(self.frequencies,
                                           combined_spectrum
                                           )
                
                
                return function_interp(freq)

            fit_function = fit_gain_factor
        elif mode == "electronic_temp":
            ice_spectrum = self.sim_spectra[0][channel_id]
            electronic_spectrum = self.sim_spectra[1][channel_id]
            galactic_spectrum = self.sim_spectra[2][channel_id]
            def fit_electronic_temp(freq, gain, el_ampl, el_cst, f0):
                # Johnson-Nyquist: V² ~ T

#                ch_bandpass = channelBandPassFilter()
#                filt = ch_bandpass.get_filter(self.frequencies, station_id=-1, channel_id=-1, det=-1, passband=self.bandpass, filter_type=self.bandpass_type, order=10)
#                filt = np.abs(filt)
                weight = el_ampl * (self.frequencies - f0) + el_cst
                electronic_spectrum_fit = electronic_spectrum * weight
                function_interp = interp1d(self.frequencies,
                                           gain * (ice_spectrum 
                                                   + electronic_spectrum_fit
                                                   + galactic_spectrum
                                                   )
                                           )
                
                
                return function_interp(freq)
                
            fit_function = fit_electronic_temp
        elif mode == "electronic_temp_cross":
            ice_spectrum = self.sim_spectra[0][channel_id]
            electronic_spectrum = self.sim_spectra[1][channel_id]
            galactic_spectrum = self.sim_spectra[2][channel_id]

            ice_el_cross = self.cross_products[0][channel_id]
            ice_gal_cross = self.cross_products[1][channel_id]
            el_gal_cross = self.cross_products[2][channel_id]

            def fit_electronic_temp(freq, gain, el_ampl, el_cst, f0):
                # Johnson-Nyquist: V² ~ T

#                ch_bandpass = channelBandPassFilter()
#                filt = ch_bandpass.get_filter(self.frequencies, station_id=-1, channel_id=-1, det=-1, passband=self.bandpass, filter_type=self.bandpass_type, order=10)
#                filt = np.abs(filt)
                weight = el_ampl * (self.frequencies - f0) + el_cst

                combined_spectrum = \
                        gain * np.sqrt( \
                        (ice_spectrum)**2 + weight**2 * electronic_spectrum**2 + galactic_spectrum**2 
                    +   weight * ice_el_cross + ice_gal_cross + weight * el_gal_cross ) 

                function_interp = interp1d(self.frequencies,
                                           combined_spectrum
                                           )
                
                
                return function_interp(freq)
                
            fit_function = fit_electronic_temp
        else:
            raise NotImplementedError

        return fit_function








if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d",
                        default="/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season2023/station11/clean/average_ft_combined.pickle",
                        help="path to data pickle file")
    parser.add_argument("--sims", "-s",
                        nargs="+",
                        default=["/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/ice/station11/clean/average_ft.pickle",
                        "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/electronic/station11/clean/average_ft.pickle",
                        "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/galactic/station11/clean/average_ft.pickle"],
                        help="path to sim pickle files")
    args = parser.parse_args()

    with open(args.data, "rb") as data_file:
        data_dic = pickle.load(data_file)


    spectrum_fitter = spectrumFitter(args.data, args.sims)
    results = spectrum_fitter.get_fit_gain()
    spectrum_fitter.save_fit_results()
