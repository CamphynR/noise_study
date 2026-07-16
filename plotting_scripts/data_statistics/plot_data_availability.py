import matplotlib.pyplot as plt
import numpy as np
import os

from rnog_data.runtable import RunTable









if __name__ == "__main__":
    
    seasons = [2022, 2023, 2024]
    station_ids = [11, 12, 13, 21, 22, 23, 24]

    rt = RunTable()
    runtable_kwargs = dict(
            stations=station_ids,
            start_time=f"{seasons[0]}-01-01",
            stop_time=f"{seasons[-1]}-08-1",
            run_types=["physics"]
            )


    table = rt.get_table(**runtable_kwargs)

    forced_trigger_idx = table["trigger_soft_enabled"] == 1
    table = table[forced_trigger_idx]

    max_trigger_idx = table["trigger_rate"] < 2
    table_excluded = table[~max_trigger_idx]
    table = table[max_trigger_idx]





    data_dir_old = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.1/"
    data_dir = "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.2/"


    run_nrs = []
    for station_id in station_ids:
        run_nrs_st = []
        for season in seasons:
            try:
                if season == 2024:
                    season_str = "2024_radiant_v2"
                else:
                    season_str = str(season)
                run_nrs_st_season = os.listdir(os.path.join(data_dir,
                                                     f"season{season_str}",
                                                     f"station{station_id}",
                                                     "clean")
                                       ) 
            except FileNotFoundError:
                run_nrs_st_season = []
            run_nrs_st_season = [int(r.split("run")[1].split(".")[0]) for r in run_nrs_st_season]
            run_nrs_st.extend(run_nrs_st_season)
        run_nrs.append(np.array(sorted(run_nrs_st)))



    
    
    plt.style.use("retro")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(len(station_ids), 1, sharex=True)
    axs = np.ndarray.flatten(axs)
    for station_id in station_ids:
        ax = axs[station_ids.index(station_id)]
        table_station = table[table["station"] == station_id]

        run_nrs_station = run_nrs[station_ids.index(station_id)]
        data_times = table_station[table_station["run"].isin(run_nrs_station)]["time_start"]

        ax.bar(table_station["time_start"],
               1,
               width=np.datetime64(2, "h"),
               align="edge",
               label="runtable",
               color=colors[0]
               )
            
        ax.bar(data_times,
               0.5,
               width=np.datetime64(2, "h"),
               align="edge",
               label="data",
               color=colors[1]
               )

#        ax.scatter(table_excluded[table_excluded["station"]==station_id]["time_start"], np.ones_like(table_excluded[table_excluded["station"]==station_id]["time_start"], dtype=float),
#                   color="green",
#                   label="max trigger > 2")

        ax.set_yticklabels([])
        ax.set_ylabel(station_id, rotation=0, labelpad=15)

    ax.legend(loc="lower left", bbox_to_anchor=(1., 0.))



    fig.text(0., 0.5, 'Station', va='center', rotation='vertical')
    fig.tight_layout()
    fig.savefig("figures/data_availability")

