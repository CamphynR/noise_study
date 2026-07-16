import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



if __name__ == "__main__":
    
    season = 2023
    station_id = 11
    calibration_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/absolute_amplitude_results/season{season}/station{station_id}"
    calibration_sim_batch = [os.path.join(calibration_dir,
                                          sim_batch,
                                          f"absolute_amplitude_calibration_season{season}_st{station_id}_{sim_batch}_best_fit.csv") 
                             for sim_batch in os.listdir(calibration_dir) if sim_batch.startswith("sim")]
    calibration = [pd.read_csv(cal_path) for cal_path in calibration_sim_batch]


    channel_ids = np.arange(24)

    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(6, 4, figsize=(20, 10))
    axs = np.ndarray.flatten(axs)
    for channel_id in channel_ids:
        gains_channel = [cal["gain"][channel_id] for cal in calibration]
        axs[channel_id].hist(gains_channel)
        axs[channel_id].set_title(f"channel {channel_id}")

    fig.tight_layout()
    fig.savefig("figures/statistics/gain_spread.png")
