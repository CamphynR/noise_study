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
    station_ids = [11]
    channel_ids = np.arange(24)
    shifts = [50, -50]

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
        


        for shift in shifts:
            calibration_dir_tmp = os.path.join(calibration_dir,
                                               f"station{station_id}",
                                               f"antenna_model_vpol_v3_shift_{shift}_MHz")
            calibration_path = os.path.join(calibration_dir_tmp,
                                            f"absolute_amplitude_calibration_season{season}_st{station_id}_antenna_model_vpol_v3_shift_{shift}_MHz_best_fit.csv")
            calibration_tmp = pd.read_csv(calibration_path)
            calibration_st[f"shift_{shift}"] = calibration_tmp
                
            data_path = os.path.join(calibration_dir_tmp,
                                     f"absolute_amplitude_calibration_season{season}_st{station_id}_antenna_model_vpol_v3_shift_{shift}_MHz_plot_data.pickle")
            data_tmp = read_pickle(data_path)
            data_st[f"shift_{shift}"] = data_tmp
                    
        calibration.append(calibration_st)
        data.append(data_st)




    plt.style.use("retro")
    for station_idx, station_id in enumerate(station_ids):
        fig, ax = plt.subplots()
        ax.scatter(channel_ids, calibration[station_idx]["best_fit"]["gain"], label="default")
        for shift in shifts:
            ax.scatter(channel_ids, calibration[station_idx][f"shift_{shift}"]["gain"], label=f"{shift} MHz shift")
        ax.set_xticks(channel_ids)    
        ax.legend()
        fig.savefig(f"figures/antenna_tests/gain_scatter_plot_antenna_model_freq_shift_season{season}_st_{station_id}")


    for station_idx, station_id in enumerate(station_ids):
        pdf_path = f"figures/antenna_tests/data_sim_spectra_antenna_model_shifts_season{season}_st{station_id}.pdf"   
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
                    label = f"default (G = {convert_to_db(calibration[station_idx]['best_fit']['gain'][channel_id]):.2f} dB,\n{calibration[station_idx]['best_fit']['best_fit_template'][channel_id]})"
                    )
            for shift in shifts:
                ax.plot(
                        data[station_idx][f"shift_{shift}"]["frequencies"],
                        data[station_idx][f"shift_{shift}"]["sim"][channel_id],
                        label = f"shift {shift} MHz (G = {convert_to_db(calibration[station_idx][f'shift_{shift}']['gain'][channel_id]):.2f} dB,\n{calibration[station_idx][f'shift_{shift}']['best_fit_template'][channel_id]})"
                        )
            ax.set_xlabel("frequencies / GHz")
            ax.set_ylabel("Amplitude / V")
            ax.set_xlim(0., 1.)
            ax.legend()
            ax.set_title(f"channel {channel_id}")
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


        
    antenna_type_groups = {"deep_vpols" : [0, 1, 2, 3, 9, 10, 22, 23], "shallow_vpols" : [5, 6, 7]}


    shifts = [f"shift_{shift}" for shift in copy.copy(shifts)]
    
    fig, axs = plt.subplots(len(antenna_type_groups), 1)
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        dG = []
        gof = {}
        for shift in [*shifts, "best_fit"]:
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
                                             data[station_idx][shift]["sim"][channel_id],
                                             freq_range=fit_evaluation_range)
                    gof_st.append(gof_tmp)
                    if shift == "best_fit" and gof_tmp > 0.08:
                        print(station_id)
                        print(channel_id)
                    if shift == "best_fit":
                        continue

                    dG_tmp = calibration[station_idx][shift]["gain"][channel_id] - calibration[station_idx]["best_fit"]["gain"][channel_id]
                    if use_relative_difference:
                        dG_tmp = (dG_tmp / calibration[station_idx]["best_fit"]["gain"][channel_id]) * 100
                    dG_tmp = np.abs(dG_tmp)
                    dG_st.append(dG_tmp)
                    if dG_tmp > 500:
                        print("Warning this dG is probably bad data somewhere")
                        print("------------")
                        print(f"n = {n}")
                        print(f"station = {station_id}")
                        print(f"channel = {channel_id}")

                dG_shift.append(dG_st)
                gof_shift.append(gof_st)
            dG.append(dG_shift)
            gof[shift] = gof_shift
        # dims: [n, station, channel]
        dG = ak.Array(dG)
        bin_contents, bin_edges = np.histogram(ak.ravel(dG), bins=20)

        
        
        for shift_i, shift in enumerate(shifts):
            axs[ax_i].hist(ak.ravel(dG[shift_i]),
                           histtype="stepfilled",
                           bins=bin_edges,
                           label=f"shift {shift} MHz",
                           facecolor=colors_face[shift_i], edgecolor=colors_edge[shift_i],
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
    fig.savefig(f"figures/antenna_tests/antenna_shift_effect_season{season}.png")
    plt.close(fig)

    fig, axs = plt.subplots(len(antenna_type_groups), 1)
    axs = np.ndarray.flatten(axs)
    for ax_i, (antenna_type_group, channel_ids) in enumerate(antenna_type_groups.items()):
        _, bin_edges = np.histogram(ak.ravel(ak.Array(gof.values())), bins=20)
        for shift_i, shift in enumerate(shifts):
            axs[ax_i].hist(ak.ravel(ak.Array(gof[shift])),
                           bins=bin_edges,
                           histtype="stepfilled",
                           label=f"shift = {shift} 50 MHz",
                           facecolor=colors_face[shift_i], edgecolor=colors_edge[shift_i],
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
    fig.savefig(f"figures/antenna_tests/antenna_shift_effect_season{season}_gof.png")
    plt.close(fig)                                 
