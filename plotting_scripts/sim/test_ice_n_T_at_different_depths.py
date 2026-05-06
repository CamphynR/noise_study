import copy
import datetime
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.detector.RNO_G.rnog_detector_mod import ModDetector
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.utilities import units

from modules.channelThermalNoiseAdderFreqDependent import channelThermalNoiseAdderFreqDependent



if __name__ == "__main__":

    # SETTINGS
    station_id = 11
    nr_samples = 2048
    sampling_rate = 3.2 * units.GHz
    frequencies = np.fft.rfftfreq(nr_samples, d=1./sampling_rate)
    channel_ids = [0, 7]
    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_n1.74",
                      "HPol" : "RNOG_hpol_v4_8inch_center_n1.74"}
#    antenna_models = {"VPol" : "RNOG_vpol_v3_5inch_center_IGLU_n1.74",
#                      "HPol" : "RNOG_hpol_v4_8inch_center_IGLU_n1.74"}
    channel_types = {"VPol" : [0, 1, 2, 3, 5, 6, 9, 10, 22, 23],
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


    test_theta = 15 * units.degree
    t0 = datetime.datetime.now()
    spectra_default = []
    nr_it = 1000
    thermal_noise_adder = channelThermalNoiseAdderFreqDependent()
    thermal_noise_adder.begin(sim_library_dir="sim/library", debug=False, theta=test_theta)
    for i in range(nr_it):
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)

        thermal_noise_adder.run(event, station, detector, use_flat_temp=True)

        station = event.get_station()
        spectra_ch = []
        for channel_id in channel_ids:
            channel = station.get_channel(channel_id)
            spectra_ch.append(np.abs(channel.get_frequency_spectrum()))
        spectra_default.append(spectra_ch)
    t1 = datetime.datetime.now()
    print(f"took {t1 - t0} to run {nr_it} iterations")

    antenna_models_default = []
    for channel_id in channel_ids:
        antenna_models_default.append(detector.get_antenna_model(station_id, channel_id))

    del detector
    del station
    del channel
    del thermal_noise_adder



    # NOW DO THE SAME BUT DIFFERENT ANTENNAS
    detector = ModDetector(database_connection='RNOG_public', log_level=logging.NOTSET,
                           select_stations=station_id)
    detector.update(detector_time)

    for channel_id in channel_ids:
        if channel_id == 0:
            antenna_model = "RNOG_vpol_new_v4_n1.70"
            detector.modify_channel_description(station_id, channel_id, ["signal_chain", "VEL"], antenna_model)
        if channel_id == 7:
            antenna_model = "RNOG_vpol_new_v4_n1.60"
            detector.modify_channel_description(station_id, channel_id, ["signal_chain", "VEL"], antenna_model)


    t0 = datetime.datetime.now()
    spectra_test = []

    thermal_noise_adder = channelThermalNoiseAdderFreqDependent()
    thermal_noise_adder.begin(sim_library_dir="sim/library", debug=False, theta=test_theta)
    for i in range(nr_it):
        event = Event(run_number=-1, event_id=-1)
        station = Station(station_id)
        station.set_station_time(detector.get_detector_time())
        for channel_id in channel_ids:
            channel = Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros_like(frequencies, dtype=np.complex128), sampling_rate)
            station.add_channel(channel)
        event.set_station(station)

        thermal_noise_adder.run(event, station, detector, use_flat_temp=True)

        station = event.get_station()
        spectra_ch = []
        for channel_id in channel_ids:
            channel = station.get_channel(channel_id)
            spectra_ch.append(np.abs(channel.get_frequency_spectrum()))
        spectra_test.append(spectra_ch)
    t1 = datetime.datetime.now()
    print(f"took {t1 - t0} to run {nr_it} iterations")


    antenna_models_test = []
    for channel_id in channel_ids:
        antenna_models_test.append(detector.get_antenna_model(station_id, channel_id))

    

    plt.style.use("retro")
    pdf = PdfPages("figures/tests/test_thermal_noise_different_ant_models.pdf")
    for i, channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots()
        ax.plot(channel.get_frequencies(), np.mean(spectra_default, axis=0)[i],
                label=f"{antenna_models_default[i]}")
        ax.plot(channel.get_frequencies(), np.mean(spectra_test, axis=0)[i],
                label=f"{antenna_models_test[i]}")
        ax.set_xlabel("frequencies / GHz")
        ax.set_ylabel("spectrum at antenna / V")
        ax.legend()
        ax_title = f"Channel {channel_id}"
        if test_theta is not None:
            ax_title += f" zenith {test_theta/units.degree} degrees"
        ax.set_title(ax_title)
        ax.set_xlim(0., 1.)
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()
