import argparse
import datetime
import healpy
import matplotlib.pyplot as plt
import numpy as np
from pygdsm import GlobalSkyModel
import scipy
from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.modules.channelGalacticNoiseAdder import get_local_coordinates
from NuRadioReco.utilities import units

from utilities.utility_functions import create_sim_event


parser = argparse.ArgumentParser()
parser.add_argument("--station", "-s", type=int, default=23)
parser.add_argument("--channel", "-c", type=int, default=0)
args = parser.parse_args()

nside = 16
solid_angle = healpy.pixelfunc.nside2pixarea(nside, degrees=False)
n_samples = 2048
sampling_rate = 3.2 * units.GHz
frequencies = np.fft.rfftfreq(n_samples, d=1./sampling_rate)
freq_range = np.array([10, 1000]) * units.MHz

det = Detector(source="rnog_mongo", select_stations=args.station)
det.update(datetime.datetime(2023,8,1))

interpolation_frequencies = np.around(np.logspace(*np.log10(freq_range), num=15), 3)
sky_model = GlobalSkyModel(freq_unit="MHz")
noise_temperatures = np.zeros(
    (len(interpolation_frequencies), healpy.pixelfunc.nside2npix(nside))
)
for i_freq, noise_freq in enumerate(interpolation_frequencies):
    radio_sky = sky_model.generate(noise_freq / units.MHz)
    radio_sky = healpy.pixelfunc.ud_grade(radio_sky, nside)
    noise_temperatures[i_freq] = radio_sky


freqs = frequencies
d_f = freqs[2] - freqs[1]
c_vac = scipy.constants.c * units.m / units.s
passband = [0.01, 1.]
passband_filter = (freqs > passband[0]) & (freqs < passband[1])

times = [datetime.datetime(2023, 8, 12, i) for i in np.linspace(0, 23, 24, dtype=int)]
efield_maxs = []
for time in times:
    # coordinates Summit Station
    local_coordinates = get_local_coordinates((72.574414869,-38.4555098446), time, nside)
    
    total_efield = np.zeros_like(freqs)
    for i_pixel in range(healpy.pixelfunc.nside2npix(nside)):
        azimuth = local_coordinates[i_pixel].az.rad
        zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

        if zenith > 90. * units.deg:
                continue

        

        temperature_interpolator = scipy.interpolate.interp1d(
             interpolation_frequencies, np.log10(noise_temperatures[:, i_pixel]), kind='quadratic')
        noise_temperature = np.power(10, temperature_interpolator(freqs[passband_filter]))

        # calculate spectral radiance of radio signal using rayleigh-jeans law
        spectral_radiance = (2. * (scipy.constants.Boltzmann * units.joule / units.kelvin)
            * freqs[passband_filter] ** 2 * noise_temperature * solid_angle / c_vac ** 2)
        spectral_radiance[np.isnan(spectral_radiance)] = 0

        # calculate radiance per energy bin
        spectral_radiance_per_bin = spectral_radiance * d_f

        # calculate electric field per frequency bin from the radiance per bin
        efield_amplitude = np.sqrt(
            spectral_radiance_per_bin / (c_vac * scipy.constants.epsilon_0 * (
                    units.coulomb / units.V / units.m))) / d_f
        
        total_efield[passband_filter] += efield_amplitude
    efield_maxs.append(np.max(total_efield))

plt.plot(efield_maxs)
plt.xticks(range(len(efield_maxs)), [t.hour for t in times])
plt.show()