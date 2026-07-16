import awkward as ak
import datetime
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import units

from utilities.utility_functions import convert_db_to_amplitude






if __name__ == "__main__":
    season = 2023
    station_id = 11
    run_nr = 1966
    channel_ids = [0, 1, 2, 3]

    data_path = f"cal_pulser/data/station{station_id}/run{run_nr}/max_A.json"


    with open(data_path, "r") as file:
        data = json.load(file)

    event_dts = data["event_dts"]
    max_A = data["max_amplitudes"]

    attenuation_times = data["attenuation_times"]
    attenuation_ifo_time = data["attenuation_ifo_time"]


    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(event_dts["0"], max_A["0"], s=2)
    axs[0].set_xlabel("event time / s")
    axs[0].set_ylabel("peak to peak / V")
    axs[1].scatter(attenuation_times, attenuation_ifo_time)
    axs[1].set_xlabel("attenuation times")
    axs[1].set_ylabel("attenuation code")
    axs[1].minorticks_on()
    axs[1].grid(which="minor")
    fig.savefig("cal_pulser/test_att.png")





    pdf_path = "figures/cal_pulser/max_A_over_time.pdf"
    pdf = PdfPages(pdf_path)

    attenuation_function = interp1d(np.array(attenuation_times),
                                    convert_db_to_amplitude(np.array(attenuation_ifo_time)),
                                    bounds_error=False,
                                    fill_value="extrapolate")

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        ax.scatter(np.array(event_dts[str(channel_id)]) / units.s, max_A[str(channel_id)]) 
        ax.scatter(np.array(event_dts[str(channel_id)]) / units.s,
                   attenuation_function(np.array(event_dts[str(channel_id)])) * max_A[str(channel_id)]) 
        ax.set_xlim(2000, 7200)
        ax.set_xlabel("time / s")
        ax.set_ylabel("max amplitude / V")

        ax_atten = ax.twinx()
        ax_atten.plot(np.array(attenuation_times) / units.s, attenuation_ifo_time, color="red")
        ax_atten.set_ylabel("attenuation / dB")

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)


    pdf.close()


    max_amplitudes_no_atten = {}
    for channel_id in channel_ids:
        max_amplitudes_no_atten[channel_id] = \
            attenuation_function(np.array(event_dts[str(channel_id)])) * max_A[str(channel_id)] 


    det = Detector(select_stations = station_id)
    det.update(datetime.datetime(2023, 8, 1))


    gains_s21 = {key : 0 for key in channel_ids}
    for channel_id in channel_ids:
        signal_chain_s11 = det.get_signal_chain_response(station_id, channel_id)
        gains_s21[channel_id] = np.max(np.abs(signal_chain_s11(np.arange(0, 1., 0.001))))


    gains_calibrated = {key : 0 for key in channel_ids}
    calibration_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
    calibration = pd.read_csv(calibration_path)
    for channel_id in channel_ids:
        gains_calibrated[channel_id] = calibration["gain"][channel_id]





    facealpha = 0.6
    colors_face = [(205./255, 146./255, 218./255, facealpha), # pink
                   (82./255, 27./255, 241./255, facealpha),   # blue
                   (190./255, 0., 0., facealpha),             # red
                   (238./255, 206./255, 0., facealpha),       # yellow
                   (33./255, 175./255, 0., facealpha),        # green
                   (110./255, 18./255, 177./255, facealpha),  # purple
                   (110./255, 183./255, 203./255, facealpha), # turquose
                   (237./255, 103./255, 40./255, facealpha),   # orange
                   (10./255, 10./255, 10./255, facealpha)   # black
                   ]

    edgealpha = 1.
    colors_edge = [(205./255, 146./255, 218./255, edgealpha), # pink
                   (82./255, 27./255, 241./255, edgealpha),   # blue
                   (190./255, 0., 0., edgealpha),             # red
                   (238./255, 206./255, 0., edgealpha),       # yellow
                   (33./255, 175./255, 0., edgealpha),        # green
                   (110./255, 18./255, 177./255, edgealpha),  # purple
                   (110./255, 183./255, 203./255, edgealpha), # turquose
                   (237./255, 103./255, 40./255, edgealpha),   # orange
                   (10./255, 10./255, 10./255, edgealpha)   # black
                   ]




    pdf_path = f"figures/cal_pulser/max_A_hist"

    station_appendix = f"_st_{station_id}"
    pdf_path += station_appendix

    pdf_path += f"_run_{run_nr}"

    pdf_path += ".pdf"

    pdf = PdfPages(pdf_path)

    fig, ax = plt.subplots()
    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes_no_atten[c] / gains_calibrated[c] for c in channel_ids])))
    plt.close(fig)
    for channel_idx, channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots()
        ax.hist(max_amplitudes_no_atten[channel_id] / gains_s21[channel_id] / units.mV,
                bins=binedges,
                label = f"channel {channel_id}, s21",
                histtype="stepfilled",
                edgecolor=(0., 0., 0., 0.),
                facecolor=colors_face[channel_idx],

#                density=True
                )
        ax.hist(max_amplitudes_no_atten[channel_id] / gains_calibrated[channel_id] / units.mV,
                bins=binedges,
                label = f"channel {channel_id}, calibrated",
                histtype="stepfilled",
                edgecolor=colors_edge[channel_idx],
                facecolor=(0., 0., 0., 0.),
#                density=True
                )
        ax.set_xlabel("max A / mV")
        ax.set_ylabel("N")
    #    ax.set_xlim(0.00025, 0.002)
    #    ax.set_ylim(0., 2700)
        ax.legend()
        fig.savefig(pdf, format="pdf")
    pdf.close()



    pdf_path = f"figures/cal_pulser/max_A_hist_channels_stacked"

    station_appendix = f"_st_{station_id}"
    pdf_path += station_appendix

    pdf_path += f"_run_{run_nr}"

    pdf_path += ".pdf"

    pdf = PdfPages(pdf_path)



    fig, ax = plt.subplots()
    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes_no_atten[c] / gains_calibrated[c] for c in channel_ids])))
    plt.close(fig)
    fig, ax = plt.subplots()
    max_amplitudes_no_atten_cal_all_channels = []
    max_amplitudes_no_atten_wocal_all_channels = []

    for channel_idx, channel_id in enumerate(channel_ids):
        max_amplitudes_no_atten_wocal_all_channels.extend(max_amplitudes_no_atten[channel_id] / gains_s21[channel_id])
        max_amplitudes_no_atten_cal_all_channels.extend(max_amplitudes_no_atten[channel_id] / gains_calibrated[channel_id])

    ax.hist(np.array(max_amplitudes_no_atten_wocal_all_channels) / units.mV,
            histtype="step",
            label="without calibration")
    ax.hist(np.array(max_amplitudes_no_atten_cal_all_channels) / units.mV,
            histtype="step",
            label="calibrated")

#    for channel_idx, channel_id in enumerate(channel_ids):
#        ax.hist(max_amplitudes_no_atten[channel_id] / gains_s21[channel_id],
#                bins=binedges,
##                label = f"channel {channel_id}, s21",
#                histtype="stepfilled",
#                edgecolor=colors_edge[channel_idx],
#                facecolor=(0., 0., 0., 0.),
##                density=True
#                )
#        ax.hist(max_amplitudes_no_atten[channel_id] / gains_calibrated[channel_id],
#                bins=binedges,
##                label = f"channel {channel_id}, calibrated",
#                label = f"channel {channel_id}",
#                histtype="stepfilled",
#                color=colors_edge[channel_idx],
#                facecolor=colors_face[channel_idx],
##                density=True
#                )
    ax.set_xlabel("max amplitude / mV")
    ax.set_ylabel("# cal pulser waveforms")
    ax.tick_params("x", rotation=30)
#    ax.set_xlim(0.00025, 0.002)
#    ax.set_ylim(0., 2700)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    pdf.close()
