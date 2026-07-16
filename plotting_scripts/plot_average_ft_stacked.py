import json
import matplotlib.pyplot as plt
import numpy as np
import os

from utilities.utility_functions import read_freq_spectrum_from_pickle


if __name__ == "__main__":



    average_ft_files = [
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station11",
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station12",
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station13",
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station21",
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station23",
            "/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season2023/station24",
            ]
    
    with open("configs/known_broken_channels.json", "r") as file:
        known_broken_channels = json.load(file)


    antenna_types = {"deep" : [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]}
    chosen_type = "deep"
    

    plt.style.use("astroparticle_physics")
    fig, ax = plt.subplots()
    i = 0
    for path in average_ft_files:
        path_tmp = os.path.join(path, "clean", "average_ft_combined.pickle")
        season = path.split("season")[1][:4]
        station_id = path.split("station")[1][:2]
        data = read_freq_spectrum_from_pickle(path_tmp)
        for channel_id in antenna_types[chosen_type]:
            if channel_id in known_broken_channels[season][station_id]:
                color = "gray"
                alpha = 0.3
                label=None
            else:
                color = "#b25da6"
                alpha = 0.8
                if i == 0:
                    label="Downhole channels"
                else:
                    label=None
            ax.plot(data["frequencies"],
                    data["spectrum"][channel_id]/np.max(data["spectrum"][channel_id]),
                    color=color,
                    alpha=alpha,
                    lw=0.6,
                    label=label)
            i+=1

    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.1)

    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"figures/average_ft_stacked_season{season}")
