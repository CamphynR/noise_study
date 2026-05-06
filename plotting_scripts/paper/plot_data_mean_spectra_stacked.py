import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os

from utilities.utility_functions import read_freq_spectrum_from_pickle












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()



    broken_channels_path = "configs/known_broken_channels.json"
    with open(broken_channels_path, "r") as file:
        known_broken_channels = json.load(file)



    season = 2023
    station_ids = [11, 12, 13, 21, 23, 24]

    data_dir = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season{season}"
    data_paths = [os.path.join(data_dir, f"station{st}", "clean", "average_ft_combined.pickle") for st in station_ids]

    data = np.array([read_freq_spectrum_from_pickle(path) for path in data_paths])

    
    channel_types = {
            "PA" : [0, 1, 2, 3],
            "helper VPol" : [5, 6, 7, 9, 10, 22, 23],
            "HPol" : [4, 8, 11, 21],
            "LPDA up" : [13, 16, 19],
            "LPDA down" : [12, 14, 15, 17, 18, 20]}



    pdf_path = "figures/paper/data_spectra_stacked.pdf"
    pdf = PdfPages(pdf_path)
    plt.style.use("astroparticle_physics")
    
    # hack to get color of first line in plot cycle
    baseline = plt.plot([0])
    plt.close()
    
    for channel_type, channel_ids in channel_types.items():
        fig, ax = plt.subplots()
        for station_idx, station_id in enumerate(station_ids):
            for channel_id in channel_ids:

                if channel_id in known_broken_channels[str(season)][str(station_id)]:
                    continue


                frequencies = data[station_idx]["frequencies"]
                spectrum = data[station_idx]["spectrum"][channel_id]
                if args.normalize:
                    spectrum /= np.max(spectrum)
                ax.plot(data[station_idx]["frequencies"], data[station_idx]["spectrum"][channel_id],
                        alpha = 0.5, color=baseline[0].get_color(), lw=1.)

        ax.set_xlim(0., 1.)
        ax.set_xlabel("frequency / GHz")
        ax.set_ylabel("amplitude / V")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf")


    pdf.close()
