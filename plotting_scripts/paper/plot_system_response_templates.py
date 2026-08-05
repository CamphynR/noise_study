import json
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import units

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator




if __name__ == "__main__":
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)  
    system_response_paths = ["sim/library/system_response_templates_deep.json",
                             "sim/library/system_response_templates_surface.json"]

    with open(system_response_paths[0], "r") as f:
        deep_keys = list(json.load(f).keys())
    deep_keys.remove("time")
    # v3_ch5 was a test and does not contain a physical template
    template_keys = deep_keys

    with open(system_response_paths[1], "r") as f:
        surface_keys = list(json.load(f).keys())
    surface_keys += ["surface_query"]
    surface_keys.remove("time")
    surface_keys.remove("v3_ch5")
    template_keys += surface_keys

    
    system_response_types = {
            "Deep (PA)" : ["ch2", "ch2_6dB"],
            "Deep (non-PA)" : ["v2_ch9", "v2_ch11"],
            "Shallow" : ["surface_query", "v2_ch13"]
            }


    plt.style.use("astroparticle_physics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["solid", "dashed"]
    figname = "figures/paper/system_response_templates.pdf"

    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    axs = np.ndarray.flatten(axs)

    frequencies = np.linspace(0, 1., 1000)
    for ax_i, (response_type, response_keys) in enumerate(system_response_types.items()):
        for response_i, response_key in enumerate(response_keys):
            system_response = systemResponseTimeDomainIncorporator()
            system_response.begin(det=0, response_path=system_response_paths, overwrite_key=response_key, bandpass_kwargs=bandpass_kwargs)
            response = system_response.get_response(channel_id=0)

            if response_i == 0:
                label = response_type
            else:
                label = None
            axs[ax_i].plot(frequencies,
                           response["gain"](frequencies),
                           lw=3.,
                           ls=linestyles[response_i],
                           color=colors[ax_i],
                           label=label
                           )
        axs[ax_i].set_xlim(0, 1.)
    axs[0].set_ylabel("response gain / a.u.", size="x-large")
    fig.text(0.5, 0., "frequency / GHz", ha="center", va="center", size="x-large")
    fig.legend(loc="center", bbox_to_anchor=(0.5, 1.02), ncols=3)
    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")

