import argparse
import csv
import datetime
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from NuRadioReco.utilities import units
from rnog_data.runtable import RunTable

from utilities.utility_functions import read_pickle





def parse_temperature_file(temp_path):
    time_temp = []
    temperature = []
    with open(temp_path, "r") as temp_file:
        reader = csv.DictReader(temp_file)
        count = -1
        time_tmp = []
        temperature_tmp = []
        for i, row in enumerate(reader):
            time_tmp_tmp = float(row["time [unix s]"])
            time_tmp.append(time_tmp_tmp)
            
            temperature_tmp_tmp = row["T_r1 [\N{DEGREE SIGN}C]"]

            # season 2022 station 24 has some empty entries
            if len(temperature_tmp_tmp) == 0:
                continue
            if float(temperature_tmp_tmp) == -64:
                # print("warning found values set to -64")
                continue

            count += 1
            temperature_tmp.append(float(temperature_tmp_tmp))



            if count == 1000:
                count = 0
#                time_median = np.datetime64(datetime.datetime.fromtimestamp(np.median(time_tmp)))
                time_median = np.median(time_tmp)
                time_temp.append(time_median)
                temperature.append(np.mean(temperature_tmp))
                time_tmp = []
                temperature_tmp = []

    temp_interpolate = interp1d(time_temp, temperature)
    return temp_interpolate


    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_appendix", default=None)
    args = parser.parse_args()
    seasons = [2022, 2023, "2024_radiant_v2"]
    station_ids = [11, 13, 23, 24]
    channel_ids = list(np.arange(24))

    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)
    

    seasons_int = []
    for season in seasons:
        if season == "2024_radiant_v2":
            seasons_int.append(2024)
        else:
            seasons_int.append(season)

    rt = RunTable()
    runtable_kwargs = dict(
            stations=station_ids,
            start_time=f"{seasons_int[0]}-01-01",
            stop_time=f"{seasons_int[-1]}-12-31",
            run_types=["physics"]
            )
    table = rt.get_table(**runtable_kwargs)

    forced_trigger_idx = table["trigger_soft_enabled"] == 1
    table = table[forced_trigger_idx]


    outlier_limits = [-0.2, 0.2]

    sensor_type = "remote1"

    temperatures = [[0 for st in station_ids] for season in seasons]

    for season in seasons:
        if season == "2024_radiant_v2":
            season_int = 2024
        else:
            season_int = season
        temperature_directory = f"station_temperatures/{sensor_type}/season{season_int}/"
        temperature_files = np.array(os.listdir(temperature_directory))
        for station_id in station_ids:

            temp_path_station_index = [f"st{station_id}" in te for te in temperature_files]
            temp_path = os.path.join(
                temperature_directory,
                temperature_files[temp_path_station_index][0]
                )

            temp_season_st = parse_temperature_file(temp_path)
            temperatures[seasons.index(season)][station_ids.index(station_id)] = temp_season_st

    print("finished parsing temp")

    



    times = [[[] for _ in station_ids] for season in seasons]
    run_nrs = [[[] for _ in station_ids] for season in seasons]
    gains_all = [[0 for _ in station_ids] for season in seasons]
    for season in seasons:

        for station_id in station_ids:
            # seasonal calibration used for these runs per season
            calibration_season_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
            calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            gain_season = calibration_season["gain"]


            cal_per_run_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}/slope_fixed_to_2023/season{season}_st{station_id}_all_runs_compiled_slope_fixed_to_2023.pickle"

            with open(cal_per_run_path, "rb") as file:
                cal_per_run = pickle.load(file)

            table_season = table[table["run"].isin(cal_per_run["run_nr"])]
            table_season_station = table_season[table_season["station"] == station_id]
            times[seasons.index(season)][station_ids.index(station_id)].extend(table_season_station["time_start"])
            run_nrs[seasons.index(season)][station_ids.index(station_id)].extend(table_season_station["run"])


            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            for channel_id in np.arange(24):
                gains_per_run[:, channel_id] -= gain_season[channel_id]
                gains_per_run[:, channel_id] /= gain_season[channel_id]
            gains_all[seasons.index(season)][station_ids.index(station_id)] = gains_per_run
    

    




    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]



    
    all_times = []
    all_gains = []
    pearson_correlation_values = np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan)
    temp_gain_slope = np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan)
    for station_id in station_ids:
        pdf_name = f"figures/temperature_correlations/temp_gain_correlation_st{station_id}.pdf"
        pdf = PdfPages(pdf_name)

        for channel_id in channel_ids:
            fig, ax = plt.subplots()

            for season in seasons:
                temp_at_time_gain = []
                # e.g. where temps were set to -64
                temp_bad_indices = []
                for t_i, t in enumerate(times[seasons.index(season)][station_ids.index(station_id)]):
                    try:
                        temp_at_time_gain.append(
                            temperatures[seasons.index(season)][station_ids.index(station_id)](np.datetime64(t, "s")))
                    except ValueError:
                        temp_at_time_gain.append(-64)
                        temp_bad_indices.append(t_i)
                temp_bad_indices_mask = np.full(len(times[seasons.index(season)][station_ids.index(station_id)]), True)
                temp_bad_indices_mask[temp_bad_indices] = False
                temp_at_time_gain = np.array(temp_at_time_gain)
            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temp_at_time_gain[temp_bad_indices_mask], gains_all[seasons.index(season)][station_ids.index(station_id)][temp_bad_indices_mask, channel_id] * 100, 
                        facecolor=colors[0] + "22",
                        edgecolor=colors[0] + "00"
                        )
                ax.set_ylim(95*outlier_limits[0],
                            105*outlier_limits[1])
                # for run_i, gain_run in enumerate(gains_all[station_ids.index(station_id)][:, channel_id]):
                #     if gain_run < -0.8:
                #         print(station_id)
                #         print(channel_id)
                #         print(gain_run)
                #         print(run_nrs[station_ids.index(station_id)][run_i])
                outlier_indices = (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] > outlier_limits[0]) \
                    & (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] < outlier_limits[1])
                outlier_indices = np.logical_and(temp_bad_indices_mask, outlier_indices)

                correlation = pearsonr(temp_at_time_gain[outlier_indices], gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])
                pearson_correlation_values[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = correlation.statistic

                # temp_gain_slope[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = \
                #     np.polyfit(temp_at_time_gain[outlier_indices],
                #                gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id],
                #                1)[0]
                

                all_times.extend(temp_at_time_gain[outlier_indices])
                all_gains.extend(gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])
                ax.set_xlabel("temperature / C")
                ax.set_ylabel("dgain / %")
                ax.set_title(f'channel {channel_id}')
            # fig.suptitle(f"Pearson correlation : {correlation.statistic:.2f}")

            fig.tight_layout()
            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)

        pdf.close()

    correlation = pearsonr(all_times, all_gains)
    print(correlation.statistic)
    print(correlation.pvalue)




    for season in seasons:
        for station_id in station_ids:
            pdf_name = f"figures/temperature_correlations/temp_gain_correlation_season{season}_st{station_id}.pdf"
            pdf = PdfPages(pdf_name)

            for channel_id in channel_ids:
                fig, ax = plt.subplots()

                temp_at_time_gain = []
                # e.g. where temps were set to -64
                temp_bad_indices = []
                for t_i, t in enumerate(times[seasons.index(season)][station_ids.index(station_id)]):
                    try:
                        temp_at_time_gain.append(
                            temperatures[seasons.index(season)][station_ids.index(station_id)](np.datetime64(t, "s")))
                    except ValueError:
                        temp_at_time_gain.append(-64)
                        temp_bad_indices.append(t_i)
                temp_bad_indices_mask = np.full(len(times[seasons.index(season)][station_ids.index(station_id)]), True)
                temp_bad_indices_mask[temp_bad_indices] = False
                temp_at_time_gain = np.array(temp_at_time_gain)
            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temp_at_time_gain[temp_bad_indices_mask], gains_all[seasons.index(season)][station_ids.index(station_id)][temp_bad_indices_mask, channel_id] * 100, 
                        facecolor=colors[0] + "22",
                        edgecolor=colors[0] + "00"
                        )
                ax.set_ylim(95*outlier_limits[0],
                            105*outlier_limits[1])
                # for run_i, gain_run in enumerate(gains_all[station_ids.index(station_id)][:, channel_id]):
                #     if gain_run < -0.8:
                #         print(station_id)
                #         print(channel_id)
                #         print(gain_run)
                #         print(run_nrs[station_ids.index(station_id)][run_i])
                outlier_indices = (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] > outlier_limits[0]) \
                    & (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] < outlier_limits[1])
                outlier_indices = np.logical_and(temp_bad_indices_mask, outlier_indices)
                correlation = pearsonr(temp_at_time_gain[outlier_indices], gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])

                ax.set_xlabel("temperature / C")
                ax.set_ylabel("dgain / %")
                ax.set_title(f'channel {channel_id}')
            
                fig.suptitle(f"Pearson correlation : {correlation.statistic:.2f}")
                fig.tight_layout()
                fig.savefig(pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)

            pdf.close()




    # ax.set_ylim(-25, 25)
        
    pdf_name = f"figures/temperature_correlations/temp_gain_correlation_all_stations.pdf"
    pdf = PdfPages(pdf_name)
    

    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        all_times = []
        all_gains = []
        for station_id in station_ids:

            for season in seasons:
                temp_at_time_gain = []
                # e.g. where temps were set to -64
                temp_bad_indices = []
                for t_i, t in enumerate(times[seasons.index(season)][station_ids.index(station_id)]):
                    try:
                        temp_at_time_gain.append(
                            temperatures[seasons.index(season)][station_ids.index(station_id)](np.datetime64(t, "s")))
                    except ValueError:
                        temp_at_time_gain.append(-64)
                        temp_bad_indices.append(t_i)
                temp_bad_indices_mask = np.full(len(times[seasons.index(season)][station_ids.index(station_id)]), True)
                temp_bad_indices_mask[temp_bad_indices] = False
                temp_at_time_gain = np.array(temp_at_time_gain)
            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temp_at_time_gain[temp_bad_indices_mask], gains_all[seasons.index(season)][station_ids.index(station_id)][temp_bad_indices_mask, channel_id] * 100, 
                        facecolor=colors[0] + "22",
                        edgecolor=colors[0] + "00"
                        )
                ax.set_ylim(95*outlier_limits[0],
                            105*outlier_limits[1])
                # for run_i, gain_run in enumerate(gains_all[station_ids.index(station_id)][:, channel_id]):
                #     if gain_run < -0.8:
                #         print(station_id)
                #         print(channel_id)
                #         print(gain_run)
                #         print(run_nrs[station_ids.index(station_id)][run_i])
                outlier_indices = (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] > outlier_limits[0]) \
                    & (gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] < outlier_limits[1])
                outlier_indices = np.logical_and(temp_bad_indices_mask, outlier_indices)
                # outlier_indices[temp_bad_indices] = False
                correlation = pearsonr(temp_at_time_gain[outlier_indices], gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])
                pearson_correlation_values[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = correlation.statistic

                all_times.extend(temp_at_time_gain[outlier_indices])
                all_gains.extend(gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])
                ax.set_xlabel("temperature / C")
                ax.set_ylabel("dgain / %")
                ax.set_title(f'channel {channel_id}')
            # fig.suptitle(f"Pearson correlation : {correlation.statistic:.2f}")

        correlation = pearsonr(all_times, all_gains)
        fig.suptitle(f"Pearson correlation : {correlation.statistic:.2f}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)

    pdf.close()
    





    pdf_name = f"figures/temperature_correlations/gains_histogram_all_stations.pdf"
    pdf = PdfPages(pdf_name)
    
    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        all_gains = []
        for season in seasons:
            for station_id in station_ids:
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue
                if station_id in [12, 22]:
                    continue
                all_gains.extend(gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] * 100)
        ax.hist(all_gains, 
                facecolor=colors[0] + "88",
                    edgecolor=colors[0],
                    histtype="stepfilled",
                    lw=3.
                    )
        ax.set_xlabel("dgain / %")
        ax.set_yscale("log")
        ax.set_title(f'channel {channel_id}')

        fig.tight_layout()
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()





    channel_types = {
        "VPol" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
        "HPol" : [4, 8, 11, 21],
        "LPDA up" : [13, 16, 19],
        "LPDA down" : [12, 14, 15, 17, 18, 20]
    }


    pdf_name = "figures/temperature_correlations/correlation_histograms.pdf"
    pdf = PdfPages(pdf_name)


    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        for season_i, season in enumerate(seasons):
            ax.hist(
                np.ndarray.flatten(pearson_correlation_values[seasons.index(season), :, channel_id]),
                histtype="stepfilled",
                facecolor=colors[season_i] + "88",
                edgecolor=colors[season_i],
                lw=3.,
                label=f"season {season}"
            )
        ax.set_xlabel("Pearson correlation")
        ax.legend()
        ax.set_title(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)


    for channel_id in channel_ids:
        fig, ax = plt.subplots()
        for st_i, station_id in enumerate(station_ids):
            broken_channels = []
            for season in seasons:
                broken_channels.extend(known_broken_channels[str(season)][str(station_id)])
            if channel_id in broken_channels:
                continue
                    
            ax.hist(
                np.ndarray.flatten(pearson_correlation_values[:, station_ids.index(station_id), channel_id]),
                histtype="stepfilled",
                facecolor=colors[st_i] + "88",
                edgecolor=colors[st_i],
                lw=3.,
                label=f"station {station_id}"
            )
        ax.set_xlabel("Pearson correlation")
        ax.legend()
        ax.set_title(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)



    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        axs[ax_i].hist(
            np.ndarray.flatten(pearson_correlation_values[:, :, channel_ids]),
            histtype="stepfilled",
            facecolor=colors[0] + "88",
            edgecolor=colors[0],
            lw=3.
        )
        axs[ax_i].set_title(channel_type)
        
    fig.text(0.5, 0., "Pearson correlation", va="center", ha="center")
    fig.suptitle("all seasons all stations")
    fig.tight_layout()
    fig.savefig(pdf,
                format="pdf",
                bbox_inches="tight")
    plt.close(fig)


    # fig, axs = plt.subplots(2, 2)
    # axs = np.ndarray.flatten(axs)
    # for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
    #     axs[ax_i].hist(
    #         np.ndarray.flatten(100* temp_gain_slope[:, :, channel_ids]),
    #         histtype="stepfilled",
    #         facecolor=colors[0] + "88",
    #         edgecolor=colors[0],
    #         lw=3.
    #     )
    #     axs[ax_i].set_title(channel_type)
        
    # fig.text(0.5, 0., "dG/dT / %/degree ", va="center", ha="center")
    # fig.suptitle("all seasons all stations")
    # fig.tight_layout()
    # fig.savefig(pdf,
    #             format="pdf",
    #             bbox_inches="tight")
    # plt.close(fig)

    pdf.close()