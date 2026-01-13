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
from utilities.utility_functions import read_pickle, find_config, read_config



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
                 fit_function_parameter_guesses=None,
                 include_impedance_mismatch_correction=False):
        """
        sim_paths : list
            list of all simulation components, without the sum
            the order is assumed to be ice, electronic, galactic, rest
        system_response : array or systemResponseTimeDomainIncorporator instance
            the module can accept a system response to apply to the simulations
        """
        config_path = find_config(sim_paths[0], sim=True)
        self.config = read_config(config_path)

    
        if include_impedance_mismatch_correction:
            impedance_mismatch_correction_path = "sim/library/impedance-matching-correction-factors.npz"
            impedance_mismatch_correction_npz = np.load(impedance_mismatch_correction_path)
            
            impedance_mismatch_correction = {}
            impedance_mismatch_correction["VPol"] = interp1d(impedance_mismatch_correction_npz["frequencies"], np.abs(impedance_mismatch_correction_npz["vpol"]), bounds_error=False, fill_value=1.) 
            impedance_mismatch_correction["HPol"] = interp1d(impedance_mismatch_correction_npz["frequencies"], np.abs(impedance_mismatch_correction_npz["hpol"]), bounds_error=False, fill_value=1.) 

        vpols = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
        hpols = [4, 8, 11, 21]
        
        self.channels_to_include = self.config["channels_to_include"]

        self.sampling_rate = sampling_rate
        self.frequencies, \
        self.data_spectrum, \
        self.var_data_spectrum, \
        self.data_header = read_freq_spec_file(data_path)

        if isinstance(system_response, systemResponseTimeDomainIncorporator):
            system_response = np.array([system_response.get_response(c)["gain"](self.frequencies) for c in range(len(self.channels_to_include))])
        

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
            if include_impedance_mismatch_correction and i != 1:
                for channel_id in vpols:
                    sim_spectrum_i[channel_id] = sim_spectrum_i[channel_id] * impedance_mismatch_correction["VPol"](sim_frequencies_i)
                    sim_var_spectrum_i[channel_id] = sim_var_spectrum_i[channel_id] * impedance_mismatch_correction["VPol"](sim_frequencies_i)**2
                for channel_id in hpols:
                    sim_spectrum_i[channel_id] = sim_spectrum_i[channel_id] * impedance_mismatch_correction["HPol"](sim_frequencies_i)
                    sim_var_spectrum_i[channel_id] = sim_var_spectrum_i[channel_id] * impedance_mismatch_correction["HPol"](sim_frequencies_i)**2
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
                self.cross_products[cross_i] = system_response * cross_product
                if include_impedance_mismatch_correction:
                    for channel_id in vpols:
                        self.cross_products[cross_i][channel_id] = self.cross_products[cross_i][channel_id] * impedance_mismatch_correction["VPol"](sim_frequencies_i)
                    for channel_id in hpols:
                        self.cross_products[cross_i][channel_id] = self.cross_products[cross_i][channel_id] * impedance_mismatch_correction["HPol"](sim_frequencies_i)

            
            for channel_id in self.channels_to_include: 
                ice_el_cross = self.cross_products[0][channel_id]
                ice_gal_cross = self.cross_products[1][channel_id]
                el_gal_cross = self.cross_products[2][channel_id]
                assert np.all(np.imag(ice_el_cross + ice_gal_cross + el_gal_cross) < 1e-20), "imaginary part of cross product sum should be 0"

                



        self.fit_range = fit_range
        self.fit_idxs = np.where(np.logical_and(fit_range[0] < self.frequencies,
                                                self.frequencies < fit_range[1]))

        self.bandpass = bandpass
        if self.bandpass is None:
            self.bandpass = [0.1, 0.7]

        self.bandpass_type = bandpass_type

        self.fit_function_parameter_guesses = fit_function_parameter_guesses
        self.cost_function = LeastSquares

        return

    def get_fit_gain(self, mode="constant", choose_channels=None):
        if choose_channels is None:
            channel_ids = self.channels_to_include
        else:
            channel_ids = choose_channels


        fit_results = []
        for channel_id in channel_ids:
            x_data = self.frequencies[self.fit_idxs]
            y_data = self.data_spectrum[channel_id][self.fit_idxs]
            y_err = self.var_data_spectrum[channel_id][self.fit_idxs]

            if mode == "constant":
                fit_function = self.select_fit_function(mode, channel_id)
                cost_function = self.cost_function(x=x_data, y=y_data,
                                             yerror=y_err,
                                             model=fit_function)
                m = Minuit(cost_function, gain=500., el_ampl=1.)
                m.limits = [(0, 1300), (0, None)]
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
                m.limits = [(0., 1500), (0., None), (0., None), (0., 0.6)]
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
                m.limits = [(300., 1500), (-10, 10.), (0., None), (0., 0.7)]
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

        return fit_results



    def save_fit_results(self, mode="electronic_temp", save_folder=None, filename=None, extended=True):
        """
        If extended is True, save all fit parameters,
        otherwise only save gain

        Also returns the fit results to use in plotting
        """
        if save_folder is None:
            save_folder = os.path.dirname(__file__)

        fit_results = self.get_fit_gain(mode=mode)

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

        return fit_results
        



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
            def fit_gain_factor(freq, gain, el_ampl):
#                weight = el_ampl * (self.frequencies - f0) + el_cst
                weight = el_ampl

                combined_spectrum = \
                        gain * np.sqrt( \
                        ice_spectrum**2 + weight**2 * electronic_spectrum**2 + galactic_spectrum**2 
                    +   weight * ice_el_cross + ice_gal_cross + weight* el_gal_cross ) 

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
                        ice_spectrum**2 + weight**2 * electronic_spectrum**2 + galactic_spectrum**2 
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
