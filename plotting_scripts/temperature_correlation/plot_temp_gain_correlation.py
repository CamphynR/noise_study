"""
Script plotting parameters related to gain temperature correltation
Default normalization uses gain and temperature per period.
if args.debug the script also plots the per season normalization
"""
import argparse
import copy
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


def pd_timestamp_to_unix(pd_timestamp):
    # as per padnas docs
    return (pd_timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s") 


def parse_temperature_file(temp_path, run_times, tolerance=2*60*60, sensor_type=None, debug=False):
    if sensor_type is None:
        sensor_type = "remote1"
    sensor_keys = {"remote1" : "T_r1 [\N{DEGREE SIGN}C]"}
    sensor_key = sensor_keys[sensor_type]

    df = pd.read_csv(temp_path)
    df["time [unix s]"] =  df["time [unix s]"].astype("int")
    run_times = pd_timestamp_to_unix(run_times)
    temperatures_tmp = []
    run_indices_skipped = []
    for run_i, run_time in enumerate(run_times):
        df_i = df.iloc[np.argmin(np.abs(df["time [unix s]"] - run_time)), :]
        if np.abs(df_i["time [unix s]"] - run_time) > tolerance:
            run_indices_skipped.append(run_i)
            continue
        temperatures_tmp.append(df_i[sensor_key])
    
    temperatures_tmp = np.array(temperatures_tmp)
    if debug:
        print(f"{len(run_indices_skipped)} runs out of {len(run_times)} skipped")

    run_indices_skipped = np.array(run_indices_skipped)

    return temperatures_tmp, run_indices_skipped



def construct_gain_per_period(run_nrs, gains, stable_periods):
    """
    gains: list of gains, assumed to be gains of one season and one station
    stable_periods: also assumed to be of one season one station
    """

    gains_copy = copy.copy(gains)

    runs_ordered_by_period= []
    gains_ordered_by_period = []
    for channel_id, gains_ch in enumerate(gains_copy):
        stable_period_indices = np.where(np.isin(run_nrs, stable_periods[str(channel_id)]))[0]
        run_nrs_per_period_ch = np.split(run_nrs, stable_period_indices)
        runs_ordered_by_period.append(run_nrs_per_period_ch)
        gains_per_period_ch = np.split(gains_ch, stable_period_indices)
        gains_ordered_by_period.append(gains_per_period_ch)
    return runs_ordered_by_period, gains_ordered_by_period

    


def is_station_in_season(season, station_id):
    if season == 2023:
        if station_id in [22]:
            return False
    if season == "2024_radiant_v2":
        if station_id in [12, 21, 22]:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_appendix", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    seasons = [2022, 2023, "2024_radiant_v2"]
#    seasons = [2023]
    station_ids = [11, 12, 13, 21, 22, 23]
#    station_ids = [21]
    channel_ids = list(np.arange(24))


    cal_per_run_basedir = "/home/ruben/Documents/data/noise_study/absolute_amplitude_results"


    outlier_limits = [-1., 1.]

    sensor_type = "remote1"



    known_broken_channels_path = "configs/known_broken_channels.json"
    with open(known_broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)

    stable_periods_path = "configs/run_gain_periods.json"
    with open(stable_periods_path, "r") as file:
        stable_periods_dict = json.load(file)
   


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



#    temperatures = [[0 for st in station_ids] for season in seasons]
#
#    for season in seasons:
#        if season == "2024_radiant_v2":
#            season_int = 2024
#        else:
#            season_int = season
#        temperature_directory = f"station_temperatures/{sensor_type}/season{season_int}/"
#        temperature_files = np.array(os.listdir(temperature_directory))
#        for station_id in station_ids:
#
#            temp_path_station_index = [f"st{station_id}" in te for te in temperature_files]
#            temp_path = os.path.join(
#                temperature_directory,
#                temperature_files[temp_path_station_index][0]
#                )
#
#            temp_season_st = parse_temperature_file(temp_path)
#            temperatures[seasons.index(season)][station_ids.index(station_id)] = temp_season_st
#
#    if args.debug:
#        print("finished parsing temp")

    



    temperatures = [[0 for st in station_ids] for season in seasons]
    times = [[[] for _ in station_ids] for season in seasons]
    run_nrs = [[[] for _ in station_ids] for season in seasons]
    run_nrs_ordered = [[[] for _ in station_ids] for season in seasons]
    gains_all = [[0 for _ in station_ids] for season in seasons]

    if args.debug:
        gains_all_norm_season = [[0 for _ in station_ids] for season in seasons]

    for season in seasons:
        if args.debug:
            print(season)

        if season == "2024_radiant_v2":
            season_int = 2024
        else:
            season_int = season
        temperature_directory = f"station_temperatures/{sensor_type}/season{season_int}/"
        temperature_files = np.array(os.listdir(temperature_directory))

        for station_id in station_ids:
            if args.debug:
                print(station_id)

            if not is_station_in_season(season, station_id):
                continue

            temp_path_station_index = [f"st{station_id}" in te for te in temperature_files]
            temp_path = os.path.join(
                temperature_directory,
                temperature_files[temp_path_station_index][0]
                )

            # seasonal calibration used for these runs per season
            calibration_season_path = f"absolute_amplitude_results/season{season}/station{station_id}/default/absolute_amplitude_calibration_season{season}_st{station_id}_best_fit.csv"
            calibration_season = pd.read_csv(calibration_season_path, index_col=0)
            gain_season = calibration_season["gain"]

            stable_period_boundary_runs = stable_periods_dict[str(season)][str(station_id)]
        

            cal_per_run_path = f"{cal_per_run_basedir}/season{season}/station{station_id}/slope_fixed_to_2023/season{season}_st{station_id}_all_runs_compiled_slope_fixed_to_2023.pickle"

            with open(cal_per_run_path, "rb") as file:
                cal_per_run = pickle.load(file)

            table_season = table[table["run"].isin(cal_per_run["run_nr"])]
            table_season_station = table_season[table_season["station"] == station_id]

            
            temperatures_season_st, run_indices_skipped = parse_temperature_file(temp_path,
                                                                                 table_season_station["time_start"],
                                                                                 sensor_type=sensor_type,
                                                                                 debug=args.debug)
            temperatures[seasons.index(season)][station_ids.index(station_id)] = temperatures_season_st

            if len(run_indices_skipped) != 0:
                runs_skipped = table_season_station["run"].to_numpy()[run_indices_skipped]
                mask = table_season_station["run"].isin(runs_skipped)
                table_season_station = table_season_station[~mask]

            times[seasons.index(season)][station_ids.index(station_id)].extend(table_season_station["time_start"])
            run_nrs[seasons.index(season)][station_ids.index(station_id)].extend(table_season_station["run"])

            gains_per_run = np.array([[cal_ch["gain"].value for cal_ch in cal_run["fit_results"]] for cal_run in cal_per_run["calibration"]])
            if len(run_indices_skipped) != 0:
                gains_per_run = np.delete(gains_per_run, run_indices_skipped, axis=0)
            runs_ordered_by_period, gains_ordered_by_period = construct_gain_per_period(table_season_station["run"].to_numpy(),
                                                                                        gains_per_run.T,
                                                                                        stable_period_boundary_runs)

            run_nrs_ordered[seasons.index(season)][station_ids.index(station_id)] = runs_ordered_by_period
            for channel_id in np.arange(24):
                for period_i, gains_period_ch in enumerate(gains_ordered_by_period[channel_ids.index(channel_id)]):
                    mean_period = np.mean(gains_period_ch)
                    gains_ordered_by_period[channel_ids.index(channel_id)][period_i] -= mean_period
                    gains_ordered_by_period[channel_ids.index(channel_id)][period_i] /= mean_period
#                gains_ordered_by_period[channel_ids.index(channel_id)] = np.concatenate(gains_ordered_by_period[channel_ids.index(channel_id)])

#            gains_all[seasons.index(season)][station_ids.index(station_id)] = np.array(gains_ordered_by_period).T
            gains_all[seasons.index(season)][station_ids.index(station_id)] = gains_ordered_by_period



 
            if args.debug:

                for channel_id in np.arange(24):
                    gains_per_run[:, channel_id] -= gain_season[channel_id]
                    gains_per_run[:, channel_id] /= gain_season[channel_id]
                    gains_all_norm_season[seasons.index(season)][station_ids.index(station_id)] = gains_per_run

                plt.style.use("retro")
                pdf = PdfPages(f"figures/tests/temp_correlation_debug_season{season}_st{station_id}.pdf")
                for test_channel_id in range(24):
                    stable_period_boundary_times = table_season_station[table_season_station["run"].isin(stable_period_boundary_runs[str(test_channel_id)])]["time_start"]
                    fig, axs = plt.subplots(2, 1, sharex=True)
                    axs = np.ndarray.flatten(axs)
                    axs[0].scatter(table_season_station["time_start"], gains_per_run[:, test_channel_id],
                                   label="season norm")
                    axs[0].scatter(table_season_station["time_start"],
                                   np.concatenate(gains_ordered_by_period[test_channel_id]),
                                   label="period norm")
                    axs[0].vlines(stable_period_boundary_times, -0.2, 0.2, ls="dotted", lw=2.)
                    axs[0].set_ylabel("dGain / %")
                    axs[1].scatter(table_season_station["time_start"], temperatures[seasons.index(season)][station_ids.index(station_id)])
                    axs[1].set_ylabel("Temperature / C")

                    axs[0].legend()
                    fig.suptitle(f"channel {test_channel_id}")
                    fig.savefig(pdf, format="pdf")
                    plt.close(fig)
                pdf.close()


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]



    
#    pearson_correlation_values = np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan)
    pearson_correlation_values = [[[[] for _ in channel_ids] for _ in station_ids] for _ in seasons]
    if args.debug:
        pearson_correlation_values_norm_season = np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan)

#    temp_gain_slope = np.full((len(seasons), len(station_ids), len(channel_ids)), np.nan)
    temp_gain_slope = [[[[] for _ in channel_ids] for _ in station_ids] for _ in seasons]
    for station_id in station_ids:
        pdf_name = f"figures/temperature_correlations/temp_gain_correlation_st{station_id}.pdf"
        pdf = PdfPages(pdf_name)

        for channel_id in channel_ids:
            fig, ax = plt.subplots()

            for season in seasons:

                if not is_station_in_season(season, station_id):
                    continue
            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temperatures[seasons.index(season)][station_ids.index(station_id)],
#                           gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] * 100, 
                           np.concatenate(gains_all[seasons.index(season)][station_ids.index(station_id)][channel_id]) * 100, 
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

                if args.debug:
                    outlier_indices_norm_season = (gains_all_norm_season[seasons.index(season)][station_ids.index(station_id)][:, channel_id] > outlier_limits[0]) \
                        & (gains_all_norm_season[seasons.index(season)][station_ids.index(station_id)][:, channel_id] < outlier_limits[1])
#                    outlier_indices_norm_season = np.logical_and(temp_bad_indices_mask, outlier_indices_norm_season)
                    correlation_season = pearsonr(temperatures[seasons.index(season)][station_ids.index(station_id)][outlier_indices_norm_season],
                                           gains_all_norm_season[seasons.index(season)][station_ids.index(station_id)][outlier_indices_norm_season, channel_id])
                    pearson_correlation_values_norm_season[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = correlation_season.statistic


                for period_i, gains_period in enumerate(gains_all[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)]):
                    runs_period = np.array(run_nrs_ordered[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)][period_i])
                    runs_season_st = np.array(run_nrs[seasons.index(season)][station_ids.index(station_id)])
                    temperature_indices = np.array([r in runs_period for r in runs_season_st])

                    outlier_indices = (gains_period > outlier_limits[0])\
                            & (gains_period < outlier_limits[1])


                    if len(gains_period[outlier_indices]) < 2:
                        if args.debug:
                            print(f"skipped period containing less than 2 runs (season {season} station: {station_id})")
                        continue
                    
                    correlation = pearsonr(temperatures[seasons.index(season)][station_ids.index(station_id)][temperature_indices][outlier_indices],
                                           gains_period[outlier_indices])
                    pearson_correlation_values[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)].append(correlation.statistic)




                    if args.debug:
                        if np.abs(correlation.statistic - correlation_season.statistic) > 0.2 and channel_id in [0, 1, 2, 3]:
                            print("------------------------------------------------------------------------------------------------------")
                            print(f"big deviation in correlation for {season} station {station_id} channel {channel_id} period {period_i}")
                            print("season norm:")
                            print(correlation_season)
                            print("period norm:")
                            print(correlation)



                # temp_gain_slope[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = \
                #     np.polyfit(temp_at_time_gain[outlier_indices],
                #                gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id],
                #                1)[0]
                

                ax.set_xlabel("temperature / C")
                ax.set_ylabel("dgain / %")
                ax.set_title(f'channel {channel_id}')
            # fig.suptitle(f"Pearson correlation : {correlation.statistic:.2f}")

            fig.tight_layout()
            fig.savefig(pdf, format="pdf", bbox_inches="tight")
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
            if len(station_ids) == 1:
                if channel_id in known_broken_channels[str(season)][str(station_ids[0])]:
                    continue
            correlation_values = [p[channel_id] for p in pearson_correlation_values[seasons.index(season)]] 
            ax.hist(
                correlation_values,
                histtype="stepfilled",
                facecolor=colors[season_i] + "88",
                edgecolor=colors[season_i],
                lw=3.,
                label=f"season {season}"
            )
            ax.hist(
                    pearson_correlation_values_norm_season[seasons.index(season), :, channel_id],
                histtype="stepfilled",
                facecolor=colors[season_i] + "00",
                edgecolor="red",
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


#    for channel_id in channel_ids:
#        fig, ax = plt.subplots()
#        for st_i, station_id in enumerate(station_ids):
#            broken_channels = []
#            for season in seasons:
#                broken_channels.extend(known_broken_channels[str(season)][str(station_id)])
#            if channel_id in broken_channels:
#                continue
#
#            correlation_values = [p[channel_id] for p in pearson_correlation_values] 
#                    
#            ax.hist(
#                np.ndarray.flatten(pearson_correlation_values[:, station_ids.index(station_id), channel_id]),
#                histtype="stepfilled",
#                facecolor=colors[st_i] + "88",
#                edgecolor=colors[st_i],
#                lw=3.,
#                label=f"station {station_id}"
#            )
#        ax.set_xlabel("Pearson correlation")
#        ax.legend()
#        ax.set_title(f"channel {channel_id}")
#        fig.tight_layout()
#        fig.savefig(pdf, format="pdf",
#                    bbox_inches="tight")
#        plt.close(fig)



    fig, axs = plt.subplots(2, 2)
    axs = np.ndarray.flatten(axs)
    for ax_i, (channel_type, channel_ids) in enumerate(channel_types.items()):
        label=None
        if args.debug:
             label="period norm"
        correlation_values = [p for p_season in pearson_correlation_values for p_st in p_season for ch in channel_ids for p in p_st[ch]] 
        hist, bins, patches = axs[ax_i].hist(
            correlation_values,
            bins=5,
            histtype="stepfilled",
            facecolor=colors[0] + "88",
            edgecolor=colors[0],
            label=label,
            lw=3.,
        )
        if args.debug:
            axs[ax_i].hist(
                np.ndarray.flatten(pearson_correlation_values_norm_season[:, :, channel_ids]),
                bins=bins,
                histtype="stepfilled",
                facecolor=colors[1] + "88",
                edgecolor=colors[1],
                lw=3.,
                label="season_norm",
            )

        axs[ax_i].legend()

        axs[ax_i].set_title(channel_type)
        
    fig.text(0., 0.5, "Counts", va="center", ha="center", rotation=90)
    fig.text(0.5, 0., "Pearson correlation", va="center", ha="center")
    fig.suptitle("all seasons all stations")
    fig.tight_layout()
    fig.savefig(pdf,
                format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    pdf.close()
    exit()

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







    for season in seasons:
        for station_id in station_ids:
            if not is_station_in_season(season, station_id):
                continue

            pdf_name = f"figures/temperature_correlations/temp_gain_correlation_season{season}_st{station_id}.pdf"
            pdf = PdfPages(pdf_name)

            for channel_id in channel_ids:
                fig, ax = plt.subplots()

            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temperatures[seasons.index(season)][station_ids.index(station_id)], np.concatenate(gains_all[seasons.index(season)][station_ids.index(station_id)][channel_ids.index(channel_id)]) * 100, 
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
                correlation = pearsonr(temperatures[seasons.index(season)][station_ids.index(station_id)][outlier_indices],
                                       gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])

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
                if not is_station_in_season(season, station_id):
                    continue
            
                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue

                ax.scatter(temperatures[seasons.index(season)][station_ids.index(station_id)], gains_all[seasons.index(season)][station_ids.index(station_id)][:, channel_id] * 100, 
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
                # outlier_indices[temp_bad_indices] = False
                correlation = pearsonr(temperatures[seasons.index(season)][station_ids.index(station_id)][outlier_indices],
                                       gains_all[seasons.index(season)][station_ids.index(station_id)][outlier_indices, channel_id])
                pearson_correlation_values[seasons.index(season), station_ids.index(station_id), channel_ids.index(channel_id)] = correlation.statistic

                all_times.extend(temperatures[seasons.index(season)][station_ids.index(station_id)])
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
                if not is_station_in_season(season, station_id):
                    continue

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





