import matplotlib.pyplot as plt
import numpy as np
from modules.pulserResponse import pulserResponse
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator







if __name__ == "__main__":


    template_paths = ["sim/library/system_response_templates_deep.json", "sim/library/system_response_templates_surface.json"]
    channel_id = 0
    template_key  ="surface_query"



    response_helper = systemResponseTimeDomainIncorporator()
    response_helper.begin(template_paths, overwrite_key=template_key)
    response = response_helper.get_response(channel_id)
    frequencies = response_helper.get_frequencies()
    del response_helper



    pulser_helper = pulserResponse()
    pulser_response = pulser_helper(frequencies)
    weights = {ch : 1./pulser_response for ch in np.arange(24)}


    response_helper = systemResponseTimeDomainIncorporator()
    response_helper.begin(template_paths, overwrite_key=template_key, weights=weights)
    response_weighted = response_helper.get_response(channel_id)



    plt.style.use("gaudi")
    fig, ax = plt.subplots()
    lw=2.
    ax.plot(frequencies, response["gain"](frequencies), label="default",
            lw=lw)
    ax.plot(frequencies, response_weighted["gain"](frequencies), label="weighted",
            lw=lw,
            ls="dashed")
    ax.set_xlim(0, 1.)
    ax.set_xlabel("frequencies / GHz")
    ax.set_ylabel("amplitude / a.u.")
    ax.set_title(template_key)

    ax.legend()
    fig.savefig("figures/tests/test_applying_weights_to_response")
