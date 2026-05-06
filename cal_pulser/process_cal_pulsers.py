import awkward as ak
import argparse
import datetime
import json
import libconf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from rnog_data.runtable import RunTable



def read_cfg(run_path):
    cfg_path = os.path.join(run_path, "cfg", "acq.cfg")
    with open(cfg_path, "r") as file:
        config = libconf.load(file)
    return config




if __name__ == "__main__":
    plt.style.use("retro")
    season = 2023
    stations = [11]
    channel_ids = [0, 1, 2, 3]


    debug = False

    det = Detector(select_stations = stations)
    det.update(datetime.datetime(2023, 8, 1))

    signal_reconstructor = channelSignalReconstructor()
    signal_reconstructor.begin()


    with open(f"configs/station_v3_installation.json", "r") as file:
        station_v3_install_date = json.load(file)

    stop_time = station_v3_install_date[str(stations[0])][0]["date"]
    

    rt = RunTable()
    runtable_kwargs = dict(
            stations = stations,
            start_time = "2023-01-01 00:00:00",
            stop_time = f"{stop_time} 00:00:00",
            run_types = ["calibration"]
            
            )
    rnog_table = rt.get_table(**runtable_kwargs)
    print(rnog_table)




    select_run_number = [1966]
#    select_run_number = None
    min_snr = 6
    max_runs = 1
    attenuation_range = [10, 20]


    # select only cal on data, every second trace should be noise


    snr = {ch_id : [] for ch_id in channel_ids}
    spectra = []

    max_amplitudes = {key : [] for key in channel_ids}
    runs_processed = 0
    for row_idx, run_info in rnog_table.iterrows():
        path = run_info["path"]
        time = run_info["time_start"]
        
        if select_run_number:
            if not run_info["run"] in select_run_number:
                continue


        full_path = os.path.join(os.environ["RNO_G_DATA"], *os.path.split(path)[0:-1])

        try:
            config = read_cfg(full_path)
        except:
            continue
#        if not config["calib"]["enable_cal"]:
#            continue
#        if not attenuation_range[0] <= config["calib"]["atten"] <= attenuation_range[1]:
#            continue
#        if config["calib"]["sweep"]["enable"]:
#            continue
        runs_processed += 1

#        print(f"attenuation: {config['calib']['atten']}")

        print(full_path)

        data_provider = dataProviderRNOG()
        det.update(time)
        data_provider.begin(full_path, det=det)

        nr_traces_to_plot = 8
        fig, axs = plt.subplots(nr_traces_to_plot//4, 4, sharex=True, sharey=True)
        axs = np.ndarray.flatten(axs)


        traces_plotted = 0
        for event in data_provider.run():
            station = event.get_station()
            triggers = station.get_triggers()
            print(triggers[0])
            exit()
            continue
            signal_reconstructor.run(event, station, det, snr_only=True)
            spectra_ch = []
            for channel in station.iter_channels():
                channel_id = channel.get_id()
                if channel_id not in channel_ids:
                    continue

                snr_tmp = channel[chp.SNR]
                snr_tmp = channel[chp.SNR]["peak_2_peak_amplitude"]
                snr[channel_id].append(snr_tmp)

                trace = channel.get_trace()
                if snr_tmp < min_snr:
                    continue

                if debug and traces_plotted < nr_traces_to_plot:
                    axs[traces_plotted].plot(channel.get_times(), channel.get_trace())
                    traces_plotted += 1

                spectra_ch.append(channel.get_frequency_spectrum())
                max_amplitudes[channel_id].append(np.max(np.abs(trace)))
            spectra.append(spectra_ch)

        if debug:
            fig.text(0.5, 0., 'time / ns', ha='center')
            fig.text(0., 0.5, 'amplitude / V', va='center', rotation='vertical')
            fig.tight_layout()
            print("saving traces")
            fig.savefig("cal_pulser/test_trace")
            plt.close(fig)

            print(np.mean(np.abs(spectra), axis=0))
            plt.plot(channel.get_frequencies(), np.mean(np.abs(spectra), axis=0)[0])
            plt.xlim(0., 1.)
            plt.savefig("cal_pulser/test_spec")
            exit()

        if runs_processed >= max_runs:
            break

    print(f"runs processed: {runs_processed}")


    pdf_path = "figures/cal_pulser/SNR_distribution"

    station_appendix = f"_st_{stations[0]}"
    for st in stations[1:]:
        station_appendix += f"_{st}"
    pdf_path += station_appendix

    if select_run_number is not None:
        if len(select_run_number) == 1:
            pdf_path += f"_run_{select_run_number[0]}"

    pdf_path += ".pdf"

    pdf = PdfPages(pdf_path)
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        hist, bin_edges, patches = ax.hist(snr[channel_id])
        ax.vlines(min_snr, 0, max(hist), color="red", ls="dashed", lw=2.)
        ax.set_xlabel("SNR")
        ax.set_ylabel("N")
        ax.set_title(f"channel {channel_id}")
        ax.set_xticks(bin_edges[::2], labels=[f"{bin_edge:.1f}" for bin_edge in bin_edges][::2], rotation=45)
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")


    pdf.close()




    gains_s21 = {key : 0 for key in channel_ids}
    for channel_id in channel_ids:
        signal_chain_s11 = det.get_signal_chain_response(stations[0], channel_id)
        gains_s21[channel_id] = np.max(np.abs(signal_chain_s11(np.arange(0, 1., 0.001))))


    gains_calibrated = {key : 0 for key in channel_ids}
    calibration_path = f"absolute_amplitude_results/season{season}/station{stations[0]}/default_v3/absolute_amplitude_calibration_season{season}_st{stations[0]}_best_fit.csv"
    calibration = pd.read_csv(calibration_path)
    for channel_id in channel_ids:
        gains_calibrated[channel_id] = calibration["gain"][channel_id]



    fig, ax = plt.subplots()
    for channel_id in channel_ids:
        ax.hist(max_amplitudes[channel_id],
                label = f"channel {channel_id}",
                histtype="stepfilled")
    ax.set_xlabel("max A / V")
    ax.set_ylabel("N")
    ax.legend()
    fig.savefig("cal_pulser/test_hist")



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

    station_appendix = f"_st_{stations[0]}"
    for st in stations[1:]:
        station_appendix += f"_{st}"
    pdf_path += station_appendix

    if select_run_number is not None:
        if len(select_run_number) == 1:
            pdf_path += f"_run_{select_run_number[0]}"

    pdf_path += ".pdf"

    pdf = PdfPages(pdf_path)

    fig, ax = plt.subplots()
    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes[c] / gains_calibrated[c] for c in channel_ids])))
    plt.close(fig)
    for channel_idx, channel_id in enumerate(channel_ids):
        fig, ax = plt.subplots()
        ax.hist(max_amplitudes[channel_id] / gains_s21[channel_id],
                bins=binedges,
                label = f"channel {channel_id}, s21",
                histtype="stepfilled",
                edgecolor=colors_edge[channel_idx],
                facecolor=None,

#                density=True
                )
        ax.hist(max_amplitudes[channel_id] / gains_calibrated[channel_id],
                bins=binedges,
                label = f"channel {channel_id}, calibrated",
                histtype="stepfilled",
                color=colors_edge[channel_idx],
                facecolor=(0., 0., 0., 0.),
#                density=True
                )
        ax.set_xlabel("max A / V")
        ax.set_ylabel("N")
    #    ax.set_xlim(0.00025, 0.002)
    #    ax.set_ylim(0., 2700)
        ax.legend()
        fig.savefig(pdf, format="pdf")
    pdf.close()



    pdf_path = f"figures/cal_pulser/max_A_hist_channels_stacked.pdf"

    station_appendix = f"_st_{stations[0]}"
    for st in stations[1:]:
        station_appendix += f"_{st}"
    pdf_path += station_appendix

    if select_run_number is not None:
        if len(select_run_number) == 1:
            pdf_path += f"_run_{select_run_number[0]}"

    pdf_path += ".pdf"

    pdf = PdfPages(pdf_path)



    fig, ax = plt.subplots()
    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes[c] / gains_calibrated[c] for c in channel_ids])))
    plt.close(fig)
    fig, ax = plt.subplots()
    for channel_idx, channel_id in enumerate(channel_ids):
        ax.hist(max_amplitudes[channel_id] / gains_s21[channel_id],
                bins=binedges,
                label = f"channel {channel_id}, s21",
                histtype="stepfilled",
                edgecolor=colors_edge[channel_idx],
                facecolor=(0., 0., 0., 0.),
#                density=True
                )
        ax.hist(max_amplitudes[channel_id] / gains_calibrated[channel_id],
                bins=binedges,
                label = f"channel {channel_id}, calibrated",
                histtype="stepfilled",
                color=colors_edge[channel_idx],
                facecolor=colors_face[channel_idx],
#                density=True
                )
    ax.set_xlabel("max A / V")
    ax.set_ylabel("N")
#    ax.set_xlim(0.00025, 0.002)
#    ax.set_ylim(0., 2700)
    ax.legend()
    fig.savefig(pdf, format="pdf")
    pdf.close()
