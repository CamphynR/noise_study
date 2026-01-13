import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import fft, units



if __name__ == "__main__":
    np.random.seed(2024)

    nr_traces = 1000
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    


    spread_ice = 1.
    spread_el = 0.5
    spread_gal = 0.3
    
    ice_trace = np.random.normal(scale=spread_ice, size=(nr_traces, nr_samples))
    el_trace = np.random.normal(scale=spread_el, size=(nr_traces, nr_samples))
    gal_trace = np.random.normal(scale=spread_gal, size=(nr_traces, nr_samples))
    summed_trace = ice_trace + el_trace + gal_trace
    vrms_summed_trace = np.sqrt(np.mean(summed_trace**2, axis=0))

    print(f"meaned vrms of summed traces is {np.mean(vrms_summed_trace)}")

    freqs = fft.freqs(nr_samples, sampling_rate)
    ice_spectra = fft.time2freq(ice_trace, sampling_rate)
    el_spectra = fft.time2freq(el_trace, sampling_rate)
    gal_spectra = fft.time2freq(gal_trace, sampling_rate)
    summed_spectra = fft.time2freq(summed_trace, sampling_rate)

    plt.plot(freqs, np.abs(summed_spectra[0]), label="sum in time domain")


    spectra_combined_in_freq_domain = ice_spectra + el_spectra + gal_spectra
    trace_combined_in_freq_domain = fft.freq2time(spectra_combined_in_freq_domain, sampling_rate)
    vrms_trace_combined_in_freq_domain = np.sqrt(np.mean(trace_combined_in_freq_domain**2, axis=0))
    print(f"meaned vrms of traces combined in freq domain is {np.mean(vrms_trace_combined_in_freq_domain)}")


    mean_spectra_combined_in_freq_domain = np.sqrt(np.mean(np.abs(spectra_combined_in_freq_domain)**2, axis=0))
    trace_mean_spectra = fft.freq2time(mean_spectra_combined_in_freq_domain, sampling_rate)
    vrms_trace_mean_spectra = np.sqrt(np.mean(trace_mean_spectra**2, axis=0))

    print(f"testing something: {np.sqrt(np.mean(vrms_summed_trace**2))}")
    print(f"vrms of trace from meaned spectra is {vrms_trace_mean_spectra}")

    plt.plot(freqs, mean_spectra_combined_in_freq_domain, label="mean of combined spectra")



    
    ice_spectra_mean = np.sqrt(np.mean(np.abs(ice_spectra)**2, axis=0))
    el_spectra_mean = np.sqrt(np.mean(np.abs(el_spectra)**2, axis=0))
    gal_spectra_mean = np.sqrt(np.mean(np.abs(gal_spectra)**2, axis=0))

    ice_el_cross = np.mean(ice_spectra * np.conjugate(el_spectra) +  np.conjugate(ice_spectra) * el_spectra, axis=0)
    ice_gal_cross = np.mean(ice_spectra * np.conjugate(gal_spectra) +  np.conjugate(ice_spectra) * gal_spectra, axis=0)
    el_gal_cross = np.mean(el_spectra * np.conjugate(gal_spectra) +  np.conjugate(el_spectra) * gal_spectra, axis=0)

    spectrum_from_meaned_components = np.sqrt(ice_spectra_mean**2 + el_spectra_mean**2 + gal_spectra_mean**2 +  (ice_el_cross + ice_gal_cross + el_gal_cross)) 
    trace_from_meaned_components = fft.freq2time(spectrum_from_meaned_components, sampling_rate)
    vrms_from_meaned_components = np.sqrt(np.mean(trace_from_meaned_components**2))
    print(f"vrms from meaned components is {vrms_from_meaned_components}")

    plt.plot(freqs, spectrum_from_meaned_components, label="spectra from meaned comps")

    plt.legend()
    plt.savefig("test")
