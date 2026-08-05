"""
code uses first fit given as default to compare to
e.g. to compare same response templates
"""
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle














if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_folders", nargs="+")
    parser.add_argument("--also_plot_data_per_fit", action="store_true")
    args = parser.parse_args()

    channel_ids = np.arange(24)
    

    fit_names = []
    plot_data = []


    fit_folder = args.fit_folders[0]

    season = fit_folder.split("season")[1].split("/")[0]
    station_id = fit_folder.split("station")[1].split("/")[0]

    fit_name = fit_folder.split("/")[-1]
    fit_names.append(fit_name)
    for path in os.listdir(fit_folder):
        if "plot_data.pickle" in path:
            with open(os.path.join(fit_folder, path), "rb") as file:
                plot_data_tmp = pickle.load(file)
            plot_data.append(plot_data_tmp)
        if "best_fit.csv" in path:
            best_fit_path = path




    for fit_i, fit_folder in enumerate(args.fit_folders[1:]):
        fit_name = fit_folder.split("/")[-1]
        fit_names.append(fit_name)
        for path in os.listdir(fit_folder):
            if "plot_data.pickle" in path:
                with open(os.path.join(fit_folder, path), "rb") as file:
                    plot_data_tmp = pickle.load(file)
                plot_data.append(plot_data_tmp)
            if "best_fit.csv" in path:
                best_fit_path = path





    plt.style.use("astroparticle_physics")

    pdf_name = f"figures/fit_comparisons/compare_fit_spectra_season{season}_st{station_id}.pdf"
    pdf = PdfPages(pdf_name)

    lw = 2.

    for channel_id in channel_ids:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        axs[0].plot(plot_data[0]["frequencies"],
                plot_data[0]["data"][channel_id],
                # label="data",
                label="data 2022",
                lw=lw
                )

        axs[0].plot(plot_data[0]["frequencies"],
                plot_data[0]["sim"][channel_id],
                label=fit_names[0],
                lw=lw
                )

        res = (plot_data[0]["sim"][channel_id] - plot_data[0]["data"][channel_id]) / plot_data[0]["data"][channel_id]
        axs[1].plot(plot_data[0]["frequencies"],
                    res,
                    label=fit_names[fit_i]
        )
        axs[1].set_ylabel("residuals / %")


        for fit_i, _ in enumerate(args.fit_folders):
            if fit_i == 0:
                continue
            if args.also_plot_data_per_fit:
                axs[0].plot(plot_data[fit_i]["frequencies"],
                        plot_data[fit_i]["data"][channel_id],
                        label="data 2023",
                        lw=lw
                        )
            axs[0].plot(plot_data[fit_i ]["frequencies"],
                    plot_data[fit_i]["sim"][channel_id],
                    label=fit_names[fit_i],
                    lw=lw)
            res = (plot_data[fit_i]["sim"][channel_id] - plot_data[fit_i]["data"][channel_id]) / plot_data[fit_i]["data"][channel_id]
            axs[1].plot(plot_data[fit_i]["frequencies"],
                        res,
                        label=fit_names[fit_i]
                        )
        axs[1].set_ylim(-0.5, 0.5)
        
        for ax in axs:
            ax.legend(loc="upper left", bbox_to_anchor=(1.,1))

        axs[0].set_xlim(0., 1.)
        axs[0].set_ylabel("amplitude / V/GHz")

        axs[1].set_xlabel("frequency / GHz")

        fig.suptitle(f"channel {channel_id}")
        fig.tight_layout()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)

    pdf.close()
