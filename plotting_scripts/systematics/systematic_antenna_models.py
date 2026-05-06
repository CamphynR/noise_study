import argparse
import awkward as ak
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


from NuRadioReco.utilities import units

from utilities.utility_functions import convert_to_db




def integral_difference(frequencies, data, sim, freq_range):
    freq_mask = (freq_range[0] < frequencies) & (frequencies < freq_range[1])
    return np.trapezoid(np.abs(data[freq_mask] - sim[freq_mask]), frequencies[freq_mask])/np.trapezoid(np.abs(data[freq_mask]), frequencies[freq_mask])





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",type=int)
    parser.add_argument("--station", type=int, nargs="+")
    parser.add_argument("--gain_in_dB", action="store_true")
    parser.add_argument("--use_relative_difference", action="store_true")
    parser.add_argument("--include_antenna_max_ratios", action="store_true")
    args = parser.parse_args()

    station_ids = args.station
    indices_of_refraction = [110, 120, 130, 140, 150, 160, 170, 180]

    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    antenna_max_path = "sim/library/antenna_models_max.json"
    with open(antenna_max_path, "r") as file:
        antenna_models_max = json.load(file)

    fit_evaluation = integral_difference
    fit_evaluation_range = [0.15, 0.6]

    gain_in_dB = args.gain_in_dB
    use_relative_difference = args.use_relative_difference

    calibration_dir = f"/user/rcamphyn/noise_study/absolute_amplitude_results/season{args.season}"
    calibration = []
    data = []
    settings_have_been_read = False
    for station_id in station_ids:
        calibration_st = {}
        data_st = {}

        calibration_base = os.path.join(calibration_dir,
                                        f"station{station_id}",
                                        f"default")
        calibration_path = os.path.join(calibration_base,
                                        f"absolute_amplitude_calibration_season{args.season}_st{station_id}_best_fit.csv")

        calibration_tmp = pd.read_csv(calibration_path, index_col=0)
        # this will always include all 24 channels
        gain = calibration_tmp["gain"].to_numpy()
        if gain_in_dB:
            gain = convert_to_db(gain)
        calibration_st["best_fit"] = gain
        # we want to compare within the best fit template to isolate only the effect of the antenna
        best_fit_template_keys = calibration_tmp["best_fit_template"]

        data_path = os.path.join(calibration_base,
                                 f"absolute_amplitude_calibration_season{args.season}_st{station_id}_plot_data.pickle")
        with open(data_path, "rb") as file:
            data_tmp = pickle.load(file)
        data_st["best_fit"] = {}
        data_st["best_fit"]["frequencies"] = data_tmp["frequencies"]
        data_st["best_fit"]["data"] = data_tmp["data"]
        data_st["best_fit"]["sim"] = data_tmp["sim"]


        for n in indices_of_refraction:
            calibration_base = os.path.join(calibration_dir,
                                            f"station{station_id}",
                                            f"antenna_model_vpol_v4_n{n}")


            if not settings_have_been_read:
                with open(os.path.join(calibration_base, "fit_settings.json"), "r") as file:
                    fit_settings = json.load(file)
                settings_have_been_read = True


            data_path = os.path.join(calibration_base,
                                     f"absolute_amplitude_calibration_season{args.season}_st{station_id}_antenna_model_vpol_v4_n{n}_plot_data_all_templates.pickle")
            with open(data_path, "rb") as file:
                data_tmp_all_templates = pickle.load(file)
            data_st[n] = {}
            data_st[n]["frequencies"] = data_tmp_all_templates["frequencies"]
            data_st[n]["data"] = list(data_tmp_all_templates["data"].values())[0]


            calibration_tmp = {}
            data_tmp = {}
            for ch_idx, channel_id in enumerate(fit_settings["channels_to_include"]):
                template_key = best_fit_template_keys[channel_id]
                calibration_path = os.path.join(calibration_base,
                                                f"absolute_amplitude_calibration_season{args.season}_st{station_id}_antenna_model_vpol_v4_n{n}_key{template_key}.csv")
                gain_channel = pd.read_csv(calibration_path, index_col=0)["gain"][channel_id]
                if gain_in_dB:
                    gain_channel = convert_to_db(gain_channel)
                calibration_tmp[channel_id] = gain_channel 

                data_tmp[channel_id] = np.array(data_tmp_all_templates["sim"][template_key][ch_idx])

            calibration_st[n] = calibration_tmp
            data_st[n]["sim"] = data_tmp

        

        calibration.append(calibration_st)
        data.append(data_st)






    plt.style.use("retro")
    # HISTOGRAM OF GAIN DEVIATION FROM BEST FIT

    antenna_type_groups = {"deep_vpols" : [0, 1, 2, 3, 9, 10, 22, 23], "shallow_vpols" : [5, 6, 7]}

    facealpha = 0.2
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

    fig, axs = plt.subplots(len(antenna_type_groups), 1)
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        dG = []
        gof = {}
        for n in [*indices_of_refraction, "best_fit"]:
            dG_n = []
            gof_n = []
            for station_idx, station_id in enumerate(station_ids):
                dG_st = []
                gof_st = []
                for channel_id in channel_ids:
                    if channel_id in known_broken_channels[str(args.season)][str(station_id)]:
                        continue
                    gof_tmp = fit_evaluation(data[station_idx]["best_fit"]["frequencies"],
                                             data[station_idx]["best_fit"]["data"][channel_id],
                                             data[station_idx][n]["sim"][channel_id],
                                             freq_range=fit_evaluation_range)
                    gof_st.append(gof_tmp)
                    if n == "best_fit" and gof_tmp > 0.08:
                        print(station_id)
                        print(channel_id)
                    if n == "best_fit":
                        continue

                    dG_tmp = calibration[station_idx][n][channel_id] - calibration[station_idx]["best_fit"][channel_id]
                    if use_relative_difference:
                        dG_tmp = (dG_tmp / calibration[station_idx]["best_fit"][channel_id]) * 100
                    dG_tmp = np.abs(dG_tmp)
                    dG_st.append(dG_tmp)
                    if dG_tmp > 500:
                        print("Warning this dG is probably bad data somewhere")
                        print("------------")
                        print(f"n = {n}")
                        print(f"station = {station_id}")
                        print(f"channel = {channel_id}")

                dG_n.append(dG_st)
                gof_n.append(gof_st)
            dG.append(dG_n)
            gof[n] = gof_n
        # dims: [n, station, channel]
        dG = ak.Array(dG)
        bin_contents, bin_edges = np.histogram(ak.ravel(dG), bins=20)

        
        
        for n_i, n in enumerate(indices_of_refraction):
            axs[ax_i].hist(ak.ravel(dG[n_i]),
                           histtype="stepfilled",
                           bins=bin_edges,
                           label=f"n={str(n)[0]}.{str(n)[1]}",
                           facecolor=colors_face[n_i], edgecolor=colors_edge[n_i],
                           lw=2.)

            max_default_antenna_model = antenna_models_max["RNOG_vpol_v3_5inch_center_n1.74"]
            max_antenna_model = antenna_models_max[f"RNOG_vpol_new_v4_n{str(n)[0]}.{str(n)[1:]}"]
            ratio_max = np.abs(1 - max_antenna_model/max_default_antenna_model) * 100
            if args.include_antenna_max_ratios:
                axs[ax_i].vlines(ratio_max, 0., np.max(bin_contents), color=colors_edge[n_i], lw=3, ls="dashed", label=f"ratio 1 - antenna_n{n}/default")
                legend_loc = "upper left"
                bbox_to_anchor = (1., 1.)
            else:
                legend_loc = "upper right"
                bbox_to_anchor = (1., 1.)

        axs[ax_i].legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, ncols=2, fontsize="xx-small")
        axs[ax_i].set_title(list(antenna_type_groups.keys())[ax_i])
        xlabel = "dG"
        if use_relative_difference:
            xlabel += "[%]"
        if gain_in_dB:
            xlabel += "(dB)"
#        else:
#            xlabel += "(amplitude)"
        axs[ax_i].set_xlabel(xlabel)

        axs[ax_i].set_ylabel("N")
        
    fig.tight_layout()
    fig.savefig(f"figures/antenna_tests/antenna_effect_season{args.season}.png")
    plt.close(fig)

    fig, axs = plt.subplots(len(antenna_type_groups), 1)
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        _, bin_edges = np.histogram(ak.ravel(ak.Array(gof.values())), bins=20)
        for n_i, n in enumerate(indices_of_refraction):
            axs[ax_i].hist(ak.ravel(ak.Array(gof[n])),
                           bins=bin_edges,
                           histtype="stepfilled",
                           label=f"n={str(n)[0]}.{str(n)[1]}",
                           facecolor=colors_face[n_i], edgecolor=colors_edge[n_i],
                           lw=2.)
        axs[ax_i].hist(ak.ravel(ak.Array(gof["best_fit"])),
                       bins=bin_edges,
                       histtype="stepfilled",
                       label="default",
                       facecolor=colors_face[-1],
                       edgecolor=colors_edge[-1],
                       lw=2.)
        axs[ax_i].legend(loc="upper right", ncols=3, fontsize="x-small")
        axs[ax_i].set_title(list(antenna_type_groups.keys())[ax_i])
        axs[ax_i].set_xlabel(f"gof ({fit_evaluation.__name__})")
        axs[ax_i].set_ylabel("N")


    fig.tight_layout()
    fig.savefig(f"figures/antenna_tests/antenna_effect_season{args.season}_gof.png")
    plt.close(fig)




    # FINALLY PLOT ALL MODELS OVER EACH OTHER
    
    
    for station_idx, station_id in enumerate(station_ids):
        pdf_path = f"figures/antenna_tests/data_fit_spectra_all_antenna_models_season{args.season}_st{station_id}.pdf"
        pdf = PdfPages(pdf_path)

        for channel_id in fit_settings["channels_to_include"]:
            fig, ax = plt.subplots()
            frequencies = data[station_idx]["best_fit"]["frequencies"]
            ax.plot(frequencies,
                    data[station_idx]["best_fit"]["data"][channel_id],
                    label = "data")

            gof_tmp_default = fit_evaluation(frequencies,
                                     data[station_idx]["best_fit"]["data"][channel_id],
                                     data[station_idx]["best_fit"]["sim"][channel_id],
                                     fit_evaluation_range) 

            ax.plot(frequencies,
                    data[station_idx]["best_fit"]["sim"][channel_id],
                    lw=3,
                    ls="dashed",
                    color="darkorange",
#                    label=f"default fit (gof : {gof_tmp_default:.3f})"
                    label=f"default fit (G : {calibration[station_idx]['best_fit'][channel_id]:.2f} dB)"
                    )
            for n_i, n in enumerate(indices_of_refraction):
                gof_tmp = fit_evaluation(frequencies,
                                         data[station_idx]["best_fit"]["data"][channel_id],
                                         data[station_idx][n]["sim"][channel_id],
                                         fit_evaluation_range) 
                ax.plot(data[station_idx]["best_fit"]["frequencies"],
                        data[station_idx][n]["sim"][channel_id],
#                        label=f"v4 n={str(n)[0]}.{str(n)[1:]} (gof = {gof_tmp:.3f})"
                        label=f"v4 G={str(n)[0]}.{str(n)[1:]} (G = {calibration[station_idx][n][channel_id]:.2f} dB)"
                        )
#                if gof_tmp < gof_tmp_default:
#                    print("------------")
#                    print(f"st = {station_id}")
#                    print(f"ch = {channel_id}")
#                    print(f"n = {n}")

            ax.set_xlim(0., 1.)
            ax.set_title(f"channel {channel_id}")
            ax.legend(loc="upper right", fontsize="x-small", ncols=2)
            ax.set_xlabel("frequencies / GHz")
            ax.set_ylabel("amplitude / V")
            fig.tight_layout()
            fig.savefig(pdf, format = "pdf")
            plt.close(fig)
        pdf.close()
