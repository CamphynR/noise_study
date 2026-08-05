import awkward as ak
import copy
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle



from utilities.utility_functions import convert_to_db



def read_pickle(path):
    with open(path, "rb") as file:
        content = pickle.load(file)
    return content    




def integral_difference(frequencies, data, sim, freq_range):
    freq_mask = (freq_range[0] < frequencies) & (frequencies < freq_range[1])
    return np.trapezoid(np.abs(data[freq_mask] - sim[freq_mask]), frequencies[freq_mask])/np.trapezoid(np.abs(data[freq_mask]), frequencies[freq_mask])



if __name__ == "__main__":


    season = 2023
    station_ids = [11, 12, 13, 21, 23, 24]
    channel_ids = np.arange(24)
    gal_models = ["gsm2016_down_2.5percent", "lfss_up_7percent"]
    gal_models_nice_names = ["GSM2016", "LFSM"]

    fit_evaluation = integral_difference
    fit_evaluation_range = [0.1, 0.7]

    use_relative_difference = True


    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)


    
    calibration_dir = f"absolute_amplitude_results/season{season}"

    calibration = []
    data = []
    for station_id in station_ids:
        calibration_st = {}
        data_st = {}
        calibration_dir_tmp = os.path.join(calibration_dir,
                                           f"station{station_id}",
                                           f"default")
        calibration_path = os.path.join(calibration_dir_tmp,
                                        f"absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv")
        calibration_tmp = pd.read_csv(calibration_path)
        calibration_st["best_fit"] = calibration_tmp

        data_path = os.path.join(calibration_dir_tmp,
                                 f"absolute_amplitude_calibration_season{season}_st{station_id}_plot_data.pickle")
        data_tmp = read_pickle(data_path)
        data_st["best_fit"] = data_tmp
        


        for gal_model in gal_models:
            calibration_st[gal_model] = {}
            calibration_dir_tmp = os.path.join(calibration_dir,
                                               f"station{station_id}",
                                               f"gal_{gal_model}")

            data_path = os.path.join(calibration_dir_tmp,
                                     f"absolute_amplitude_calibration_season{season}_st{station_id}_gal_{gal_model}_plot_data_all_templates.pickle")
            with open(data_path, "rb") as file:
                data_tmp_all_templates = pickle.load(file)
            data_st[gal_model] = {}
            data_st[gal_model]["frequencies"] = data_tmp_all_templates["frequencies"]
            data_st[gal_model]["data"] = list(data_tmp_all_templates["data"].values())[0]

            calibration_tmp = []
            data_tmp = {}
            for ch_idx, channel_id in enumerate(channel_ids):
                default_template = calibration_st["best_fit"]["best_fit_template"][channel_id]
                calibration_path = os.path.join(calibration_dir_tmp,
                                                f"absolute_amplitude_calibration_season{season}_st{station_id}_gal_{gal_model}_key{default_template}.csv")
                gain_channel = pd.read_csv(calibration_path, index_col=0)["gain"][channel_id]
                calibration_tmp.append(gain_channel) 

                data_tmp[channel_id] = np.array(data_tmp_all_templates["sim"][default_template][ch_idx])

            calibration_st[gal_model]["gain"] = calibration_tmp
            data_st[gal_model]["sim"] = data_tmp
                    
        calibration.append(calibration_st)
        data.append(data_st)




    plt.style.use("astroparticle_physics")
    for station_idx, station_id in enumerate(station_ids):
        fig, ax = plt.subplots()
        ax.scatter(channel_ids, calibration[station_idx]["best_fit"]["gain"], label="default")
        for gal_model in gal_models:
            ax.scatter(channel_ids, calibration[station_idx][f"{gal_model}"]["gain"], label=f"{gal_model}")
        ax.set_xticks(channel_ids)    
        ax.legend()
        fig.savefig(f"figures/galactic_noise/gain_scatter_plot_gal_models_season{season}_st_{station_id}")


    for station_idx, station_id in enumerate(station_ids):
        pdf_path = f"figures/galactic_noise/data_sim_spectra_gal_models_season{season}_st{station_id}.pdf"   
        pdf = PdfPages(pdf_path)
        
        for channel_id in channel_ids:
            fig, ax = plt.subplots()
            ax.plot(
                    data[station_idx]["best_fit"]["frequencies"],
                    data[station_idx]["best_fit"]["data"][channel_id],
                    label = "data"
                    )

            ax.plot(
                    data[station_idx]["best_fit"]["frequencies"],
                    data[station_idx]["best_fit"]["sim"][channel_id],
                    label = f"default\n(G = {convert_to_db(calibration[station_idx]['best_fit']['gain'][channel_id]):.2f} dB,\n{calibration[station_idx]['best_fit']['best_fit_template'][channel_id]})"
                    )
            for gal_model in gal_models:
                ax.plot(
                        data[station_idx][f"{gal_model}"]["frequencies"],
                        data[station_idx][f"{gal_model}"]["sim"][channel_id],
                        label = f"{gal_model}\n(G = {convert_to_db(calibration[station_idx][f'{gal_model}']['gain'][channel_id]):.2f} dB)"
                        )
            ax.set_xlabel("frequencies / GHz")
            ax.set_ylabel("Amplitude / V")
            ax.set_xlim(0., 1.)
            ax.legend(loc="upper left", bbox_to_anchor=(1.,1.), fontsize=8)
            ax.set_title(f"channel {channel_id}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")     


        pdf.close()



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


        
    antenna_type_groups = {"deep_vpols" : [0, 1, 2, 3, 9, 10, 22, 23], "shallow_vpols" : [5, 6, 7], "donwnward LPDAs" : [12, 14, 15, 17, 18, 20], "upward LPDAs" : [13, 16, 19]}


    gal_models = [f"{gal_model}" for gal_model in copy.copy(gal_models)]
    
    fig, axs = plt.subplots(len(antenna_type_groups)//2, 2, figsize=(10, 10))
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        dG = []
        gof = {}
        for gal_model in [*gal_models, "best_fit"]:
            dG_shift = []
            gof_shift = []
            for station_idx, station_id in enumerate(station_ids):
                dG_st = []
                gof_st = []
                for channel_id in channel_ids:
                    if channel_id in known_broken_channels[str(season)][str(station_id)]:
                        continue
                    gof_tmp = fit_evaluation(data[station_idx]["best_fit"]["frequencies"],
                                             data[station_idx]["best_fit"]["data"][channel_id],
                                             data[station_idx][gal_model]["sim"][channel_id],
                                             freq_range=fit_evaluation_range)
                    gof_st.append(gof_tmp)
                    if gal_model == "best_fit" and gof_tmp > 0.08:
                        print(station_id)
                        print(channel_id)
                    if gal_model == "best_fit":
                        continue

                    dG_tmp = calibration[station_idx][gal_model]["gain"][channel_id] - calibration[station_idx]["best_fit"]["gain"][channel_id]
                    if use_relative_difference:
                        dG_tmp = (dG_tmp / calibration[station_idx]["best_fit"]["gain"][channel_id]) * 100
                    dG_tmp = np.abs(dG_tmp)
                    dG_st.append(dG_tmp)
                    if dG_tmp > 1.9:
                        print("Warning this dG is probably bad data somewhere")
                        print("------------")
                        print(f"model = {gal_model}")
                        print(f"station = {station_id}")
                        print(f"channel = {channel_id}")

                dG_shift.append(dG_st)
                gof_shift.append(gof_st)
            dG.append(dG_shift)
            gof[gal_model] = gof_shift
        # dims: [n, station, channel]
        dG = ak.Array(dG)
        bin_contents, bin_edges = np.histogram(ak.ravel(dG), bins=20)

        
        
        for model_i, gal_model in enumerate(gal_models):
            axs[ax_i].hist(ak.ravel(dG[model_i]),
                           histtype="stepfilled",
                           bins=bin_edges,
                           label=f"{gal_models_nice_names[model_i]}",
                           facecolor=colors_face[model_i], edgecolor=colors_edge[model_i],
                           lw=2.)

            legend_loc = "upper right"
            bbox_to_anchor = (1., 1.)

        axs[ax_i].legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, ncols=2, fontsize="xx-small")
        axs[ax_i].set_title(list(antenna_type_groups.keys())[ax_i])
        xlabel = "dG"
        if use_relative_difference:
            xlabel += "[%]"
#        else:
#            xlabel += "(amplitude)"
        axs[ax_i].set_xlabel(xlabel)

        axs[ax_i].set_ylabel("N")
        
    fig.tight_layout()
    fig.savefig(f"figures/galactic_noise/gal_model_effect_season{season}.png")
    plt.close(fig)

    fig, axs = plt.subplots(len(antenna_type_groups)//2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        _, bin_edges = np.histogram(ak.ravel(ak.Array(gof.values())), bins=20)
        for model_i, gal_model in enumerate(gal_models):
            axs[ax_i].hist(ak.ravel(ak.Array(gof[gal_model])),
                           bins=bin_edges,
                           histtype="stepfilled",
                           label=f"{gal_model}",
                           facecolor=colors_face[model_i], edgecolor=colors_edge[model_i],
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
    fig.savefig(f"figures/galactic_noise/gal_model_effect_season{season}_gof.png")
    plt.close(fig)                                 
