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

    theta = json_dict["theta"]
    eff_temp = json_dict["eff_temperature"]
    plt.style.use("retro")
    fig, ax = plt.subplots()
    ax.plot(np.cos(theta), eff_temp, lw = 2., label="antenna at 100 m")
    theta = json_dict_1["theta"]
    eff_temp = json_dict_1["eff_temperature"]
    ax.plot(np.cos(theta), eff_temp, lw = 2., label="antenna at 1 m")
    ax.set_ylim(230, 255)
    ax.legend()
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.2, ls="dashed")
    ax.set_xlabel("cos (zenith)")
    ax.set_ylabel("Effective temperature / K")
    fig.tight_layout()
    plt.savefig("figures/POS_ICRC/eff_temp_theta", dpi=300, bbox_inches="tight")
