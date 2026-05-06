import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder
from NuRadioReco.utilities import units

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel



if __name__ == "__main__":
    
    galactic_noise_adder = channelGalacticNoiseAdder()
    galactic_noise_adder.begin(freq_range=[10*units.MHz,
                                                1600*units.MHz],
                                    ice_model="greenland_poly5",
                                    attenuation_model="GL3",
                                    caching=True)

    # SETTINGS
    station_id = 11
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    channel_ids = [0, 4, 12, 13]
    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_IGLU_n1.74",
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



    spectra = [[] for ch in channel_ids]
    nr_it = 1
    for i in range(nr_it):
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)

        galactic_noise_adder.run(event, station, detector)

        station = event.get_station()
        for i, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            spectra[i].append(np.abs(channel.get_frequency_spectrum()))
    t1 = datetime.datetime.now()



    plt.style.use("retro")
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("figures/tests/test_galactic_noise.pdf")
    for i, channel_id in enumerate(channel_ids):
        plt.plot(channel.get_frequencies(), np.mean(spectra[i], axis=0))
        plt.title(f"channel {channel_id}")
        plt.legend()
        plt.savefig(pdf, format="pdf")
        plt.close()
    pdf.close()
