import numpy as np

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.utilities.fft import freq2time
from NuRadioReco.utilities import units

from utilities.utility_functions import read_pickle


def read_freq_spec_file(path):
    result_dictionary = read_pickle(path)
    header = result_dictionary["header"]
    frequencies = result_dictionary["freq"]
    frequency_spectrum = result_dictionary["frequency_spectrum"]
    var_frequency_spectrum = result_dictionary["var_frequency_spectrum"]
    return {"frequencies" : frequencies,
            "spectrum" : frequency_spectrum,
            "var_spectrum" : var_frequency_spectrum,
            "header" : header}

def read_freq_spectrum_from_nur(files : list, event_nr=0, channel_id=0):
    event_reader = eventReader()
    event_reader.begin(files)
    spec = []
    for event in event_reader.run():
        station = event.get_station()
        channel = station.get_channel(channel_id)
        sampling_rate = channel.get_sampling_rate()
        nr_samples = channel.get_number_of_samples()
        frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
        frequency_spectrum = channel.get_frequency_spectrum()
        spec.append(np.abs(frequency_spectrum))
        break
#    spec = np.mean(spec, axis=0)
    return frequencies, spec[0]


if __name__ == "__main__":
    sampling_rate = 3.2 * units.GHz
    station_id = 11
    channel_id = 0

    bandpass = channelBandPassFilter()
    spectra_path = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/complete_sim_traces_set_v0.1_no_system_response/station11/run0/events_ice_batch0.nur"
    freq, spec = read_freq_spectrum_from_nur(spectra_path, channel_id=channel_id)
    filt = bandpass.get_filter(freq, station_id, channel_id, det=0, passband = [0.1, 0.7], filter_type="butter", order=10)
    spec = np.abs(spec) * np.abs(filt)
    trace = freq2time(spec, sampling_rate)
    vrms = np.sqrt(np.mean(trace**2))
    print(vrms/units.mV)


    average_ft_path = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season2023/station11/clean/average_ft_run1970.pickle"
    results = read_freq_spec_file(average_ft_path)
    spec = results["spectrum"]
    spec = spec[channel_id]
    trace = freq2time(spec, sampling_rate)
    vrms = np.sqrt(np.mean(trace**2))
    print(vrms/units.mV)
    
