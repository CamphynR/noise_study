import json
import matplotlib.pyplot as plt
import numpy as np




if __name__ == "__main__":
    json_path = "/user/rcamphyn/noise_study/sim/library/eff_temperature_-100m_ntheta100_GL3.json" 
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)

    json_path = "/user/rcamphyn/noise_study/sim/library/eff_temperature_-1.0m_ntheta100_GL3.json" 
    with open(json_path, "r") as json_file:
        json_dict_1 = json.load(json_file)

    json_path = "/user/rcamphyn/noise_study/sim/library/eff_temperature_-40m_ntheta100.json" 
    with open(json_path, "r") as json_file:
        json_dict_40 = json.load(json_file)

    theta = json_dict["theta"]
    eff_temp = json_dict["eff_temperature"]



    plt.style.use("astroparticle_physics")
    fig, ax = plt.subplots()
    ax.plot(np.cos(theta), eff_temp, lw = 2., label="antenna at 100 m")
    theta = json_dict_1["theta"]
    eff_temp = json_dict_1["eff_temperature"]
    ax.plot(np.cos(theta), eff_temp, lw = 2., label="antenna at 1 m")
    eff_temp = json_dict_40["eff_temperature"]
#    ax.plot(np.cos(theta), eff_temp, lw = 2., label="antenna at 40 m")
    ax.set_ylim(230, 255)
    ax.legend()
    ax.set_xlabel("cos (incident angle)")
    ax.set_ylabel(r"$T_\mathrm{eff}$ / K")
    fig.tight_layout()
    plt.savefig("figures/paper/eff_temp_theta.eps", dpi=600, bbox_inches="tight", format="eps")
    plt.savefig("figures/paper/eff_temp_theta.png", dpi=600, bbox_inches="tight", format="png")
    plt.savefig("figures/paper/eff_temp_theta.pdf", dpi=600, bbox_inches="tight", format="pdf")
