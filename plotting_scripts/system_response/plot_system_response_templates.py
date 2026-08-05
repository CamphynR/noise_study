import json
import matplotlib.pyplot as plt
import numpy as np


from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator









if __name__ == "__main__":

    template_paths = [
            "sim/library/system_response_templates_deep.json",
            "sim/library/system_response_templates_surface.json",
            ]


    templates = {}
    with open(template_paths[0], "r") as file:
        content = json.load(file)

    keys = content.keys()
    keys.remove("time")

    with open(template_paths[1], "r") as file:
        content = json.load(file)

    keys.extend(content.keys())
    keys.remove("time")

    keys.append("surface_query")


    for key in keys:
        if key == "ch2":
            frequencies_radiant_v2 = template_helper.frequencies
        if key == "v3_ch8_52dB":
            frequencies_radiant_v3 = template_helper.frequencies

        template_helper = systemResponseTimeDomainIncorporator()
        template_helper.begin(response_paths=template_paths,
                              overwrite_key=key)
        response = template_helper.get_response(0)["gain"]
        templates[key] = response

        

    print(keys) 
    plt.style.use("astroparticle_physics")
    fig, axs = plt.subplots(1, 2)
    axs = np.ndarray.flatten(axs)

    axs[0].plot(
