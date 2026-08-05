import awkward as ak
import argparse
from astropy.time import Time
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
from NuRadioReco.utilities import units
from rnog_data.runtable import RunTable



def read_cfg(run_path):
    cfg_path = os.path.join(run_path, "cfg", "acq.cfg")
    with open(cfg_path, "r") as file:
        config = libconf.load(file)
    return config




if __name__ == "__main__":
    plt.style.use("astroparticle_physics")
    season = 2023
    stations = [24]
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




#    select_run_number = [1966]
    select_run_number = [2747]
#    select_run_number = None
    min_snr = 5
    max_runs = 1
    max_events_per_run = None
    attenuation_range = [10, 20]


    sweep_settings = {
            "start_at_attenuation" : 20
            }


    # select only cal on data, every second trace should be noise


    snr = {ch_id : [] for ch_id in channel_ids}
    spectra = []

    max_amplitudes = {key : [] for key in channel_ids}
    event_times = []
    event_dts = {key : [] for key in channel_ids}
    runs_processed = 0
    for row_idx, run_info in rnog_table.iterrows():
        path = run_info["path"]

        run_start_time = run_info["time_start"]
        run_start_time = Time(run_start_time)
        det.update(run_start_time)

        sampling_rate = det.get_sampling_frequency(station_id=run_info["station"], channel_id=0)
        number_of_samples = det.get_number_of_samples(station_id=run_info["station"], channel_id=0)
        trace_time = number_of_samples / sampling_rate
        
        if select_run_number:
            if not run_info["run"] in select_run_number:
                continue


        full_path = os.path.join(os.environ["RNO_G_DATA"], *os.path.split(path)[0:-1])

        try:
            config = read_cfg(full_path)
        except:
            continue
        # for sweeps
        if config["calib"]["sweep"]["enable"]:
            is_sweep = True
            attenuation_range = \
            [config["calib"]["sweep"]["start_atten"], \
                    config["calib"]["sweep"]["stop_atten"] ]
            attenuation_step = config["calib"]["sweep"]["atten_step"]
            attenuation_nr_steps = np.abs(np.diff(attenuation_range)) / attenuation_step
            attenuation_step_time = config["calib"]["sweep"]["step_time"] * units.s

            attenuation_to_skip = np.abs(attenuation_range[0] - sweep_settings["start_at_attenuation"])
            steps_to_skip = attenuation_to_skip / attenuation_step
            step_time_to_skip = steps_to_skip * attenuation_step_time

            attenuation_times = np.arange(0,
                                          attenuation_nr_steps * attenuation_step_time,
                                          attenuation_step_time)
#            attenuation_times = [run_start_time + TimeDelta(att_t / unit.s, format="sec") for att_t in attenuation_times]
            attenuation_ifo_time = np.arange(*attenuation_range, -1*attenuation_step)

                    
#        if not config["calib"]["enable_cal"]:
#            continue
#        if not attenuation_range[0] <= config["calib"]["atten"] <= attenuation_range[1]:
#            continue
#        if config["calib"]["sweep"]["enable"]:
#            continue
        runs_processed += 1

#        print(f"attenuation: {config['calib']['atten']}")

        print(full_path)

        if is_sweep:
            run_start_time_unix = run_start_time.unix 
            selectors = [lambda eventInfo : np.abs(eventInfo.triggerTime - run_start_time_unix) > step_time_to_skip / units.s]
        else:
            selectors = []

        data_provider = dataProviderRNOG()
        data_provider.begin(full_path, det=det, reader_kwargs={"selectors" : selectors})

        nr_traces_to_plot = 8
        fig, axs = plt.subplots(nr_traces_to_plot//4, 4, sharex=True, sharey=True)
        axs = np.ndarray.flatten(axs)


        traces_plotted = 0
        events_processed = 0
        for event in data_provider.run():
            station = event.get_station()
            event_time = station.get_station_time()
            if is_sweep:
                dt = event_time - run_start_time
                dt = dt.to_value("sec") * units.s
#                if dt < step_time_to_skip:
#                    continue

            event_times.append(event_time)
            triggers = station.get_triggers()
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
                event_dts[channel_id].append(dt)
            spectra.append(spectra_ch)

            events_processed += 1
            if max_events_per_run is not None:
                if events_processed >= max_events_per_run:
                    break

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

    print(events_processed)
    print(f"runs processed: {runs_processed}")


    # SAVING

    save_dir = f"cal_pulser/data/station{stations[0]}/run{select_run_number[0]}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "max_A.json")
    save_dict = dict(
            event_dts = event_dts,
            max_amplitudes=max_amplitudes
            )

    if is_sweep:
        save_dict["is_sweep"] = 1
        save_dict["attenuation_times"] = attenuation_times
        save_dict["attenuation_ifo_time"] = attenuation_ifo_time

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(save_path, "w") as file:
        json.dump(save_dict, file, cls=NumpyEncoder)

    


#    if is_sweep:
#        plt.close()
#        fig, ax = plt.subplots()
#        ax_atten = ax.twinx()
#        ax.scatter(np.array(event_dts[0]) / units.s, max_amplitudes[0], label="max A")
#        ax_atten.plot(attenuation_times, attenuation_ifo_time, label="attenuation")
#        ax.legend()
#        fig.savefig(f"cal_pulser/attenuation_sweep_st{stations[0]}_run{select_run_number[0]}")
#
#
#    pdf_path = "figures/cal_pulser/SNR_distribution"
#
#    station_appendix = f"_st_{stations[0]}"
#    for st in stations[1:]:
#        station_appendix += f"_{st}"
#    pdf_path += station_appendix
#
#    if select_run_number is not None:
#        if len(select_run_number) == 1:
#            pdf_path += f"_run_{select_run_number[0]}"
#
#    pdf_path += ".pdf"
#
#    pdf = PdfPages(pdf_path)
#    for channel_id in channel_ids:
#        fig, ax = plt.subplots()
#        hist, bin_edges, patches = ax.hist(snr[channel_id])
#        ax.vlines(min_snr, 0, max(hist), color="red", ls="dashed", lw=2.)
#        ax.set_xlabel("SNR")
#        ax.set_ylabel("N")
#        ax.set_title(f"channel {channel_id}")
#        ax.set_xticks(bin_edges[::2], labels=[f"{bin_edge:.1f}" for bin_edge in bin_edges][::2], rotation=45)
#        fig.tight_layout()
#        fig.savefig(pdf, format="pdf")
#
#
#    pdf.close()
#
#
#
#
#    gains_s21 = {key : 0 for key in channel_ids}
#    for channel_id in channel_ids:
#        signal_chain_s11 = det.get_signal_chain_response(stations[0], channel_id)
#        gains_s21[channel_id] = np.max(np.abs(signal_chain_s11(np.arange(0, 1., 0.001))))
#
#
#    gains_calibrated = {key : 0 for key in channel_ids}
#    calibration_path = f"absolute_amplitude_results/season{season}/station{stations[0]}/default/absolute_amplitude_calibration_season{season}_st{stations[0]}_best_fit.csv"
#    calibration = pd.read_csv(calibration_path)
#    for channel_id in channel_ids:
#        gains_calibrated[channel_id] = calibration["gain"][channel_id]
#
#
#
#    fig, ax = plt.subplots()
#    for channel_id in channel_ids:
#        ax.hist(max_amplitudes[channel_id],
#                label = f"channel {channel_id}",
#                histtype="stepfilled")
#    ax.set_xlabel("max A / V")
#    ax.set_ylabel("N")
#    ax.legend()
#    fig.savefig("cal_pulser/test_hist")
#
#
#
#    facealpha = 0.6
#    colors_face = [(205./255, 146./255, 218./255, facealpha), # pink
#                   (82./255, 27./255, 241./255, facealpha),   # blue
#                   (190./255, 0., 0., facealpha),             # red
#                   (238./255, 206./255, 0., facealpha),       # yellow
#                   (33./255, 175./255, 0., facealpha),        # green
#                   (110./255, 18./255, 177./255, facealpha),  # purple
#                   (110./255, 183./255, 203./255, facealpha), # turquose
#                   (237./255, 103./255, 40./255, facealpha),   # orange
#                   (10./255, 10./255, 10./255, facealpha)   # black
#                   ]
#
#    edgealpha = 1.
#    colors_edge = [(205./255, 146./255, 218./255, edgealpha), # pink
#                   (82./255, 27./255, 241./255, edgealpha),   # blue
#                   (190./255, 0., 0., edgealpha),             # red
#                   (238./255, 206./255, 0., edgealpha),       # yellow
#                   (33./255, 175./255, 0., edgealpha),        # green
#                   (110./255, 18./255, 177./255, edgealpha),  # purple
#                   (110./255, 183./255, 203./255, edgealpha), # turquose
#                   (237./255, 103./255, 40./255, edgealpha),   # orange
#                   (10./255, 10./255, 10./255, edgealpha)   # black
#                   ]
#
#
#
#
#    pdf_path = f"figures/cal_pulser/max_A_hist"
#
#    station_appendix = f"_st_{stations[0]}"
#    for st in stations[1:]:
#        station_appendix += f"_{st}"
#    pdf_path += station_appendix
#
#    if select_run_number is not None:
#        if len(select_run_number) == 1:
#            pdf_path += f"_run_{select_run_number[0]}"
#
#    pdf_path += ".pdf"
#
#    pdf = PdfPages(pdf_path)
#
#    fig, ax = plt.subplots()
#    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes[c] / gains_calibrated[c] for c in channel_ids])))
#    plt.close(fig)
#    for channel_idx, channel_id in enumerate(channel_ids):
#        fig, ax = plt.subplots()
#        ax.hist(max_amplitudes[channel_id] / gains_s21[channel_id],
#                bins=binedges,
#                label = f"channel {channel_id}, s21",
#                histtype="stepfilled",
#                edgecolor=colors_edge[channel_idx],
#                facecolor=None,
#
##                density=True
#                )
#        ax.hist(max_amplitudes[channel_id] / gains_calibrated[channel_id],
#                bins=binedges,
#                label = f"channel {channel_id}, calibrated",
#                histtype="stepfilled",
#                color=colors_edge[channel_idx],
#                facecolor=(0., 0., 0., 0.),
##                density=True
#                )
#        ax.set_xlabel("max A / V")
#        ax.set_ylabel("N")
#    #    ax.set_xlim(0.00025, 0.002)
#    #    ax.set_ylim(0., 2700)
#        ax.legend()
#        fig.savefig(pdf, format="pdf")
#    pdf.close()
#
#
#
#    pdf_path = f"figures/cal_pulser/max_A_hist_channels_stacked.pdf"
#
#    station_appendix = f"_st_{stations[0]}"
#    for st in stations[1:]:
#        station_appendix += f"_{st}"
#    pdf_path += station_appendix
#
#    if select_run_number is not None:
#        if len(select_run_number) == 1:
#            pdf_path += f"_run_{select_run_number[0]}"
#
#    pdf_path += ".pdf"
#
#    pdf = PdfPages(pdf_path)
#
#
#
#    fig, ax = plt.subplots()
#    bins, binedges, patches = ax.hist(ak.ravel(ak.Array([max_amplitudes[c] / gains_calibrated[c] for c in channel_ids])))
#    plt.close(fig)
#    fig, ax = plt.subplots()
#    for channel_idx, channel_id in enumerate(channel_ids):
#        ax.hist(max_amplitudes[channel_id] / gains_s21[channel_id],
#                bins=binedges,
#                label = f"channel {channel_id}, s21",
#                histtype="stepfilled",
#                edgecolor=colors_edge[channel_idx],
#                facecolor=(0., 0., 0., 0.),
##                density=True
#                )
#        ax.hist(max_amplitudes[channel_id] / gains_calibrated[channel_id],
#                bins=binedges,
#                label = f"channel {channel_id}, calibrated",
#                histtype="stepfilled",
#                color=colors_edge[channel_idx],
#                facecolor=colors_face[channel_idx],
##                density=True
#                )
#    ax.set_xlabel("max A / V")
#    ax.set_ylabel("N")
##    ax.set_xlim(0.00025, 0.002)
##    ax.set_ylim(0., 2700)
#    ax.legend()
#    fig.savefig(pdf, format="pdf")
#    pdf.close()
