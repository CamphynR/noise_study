import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator



if __name__ == "__main__":
    bandpass_kwargs = dict(passband=[0.1, 0.7], filter_type="butter", order=10)  
    system_response_paths = ["sim/library/deep_templates_combined.json",
                             "sim/library/v2_v3_surface_impulse_responses.json"]
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

    

    plt.style.use("astroparticle_physics")
    pdf_name = "figures/templates.pdf"
    pdf = PdfPages(pdf_name) 

    frequencies = np.linspace(0, 1.6, 1000)
    for template_key in template_keys:
        system_response = systemResponseTimeDomainIncorporator()
        system_response.begin(det=0, response_path=system_response_paths, overwrite_key=template_key, bandpass_kwargs=bandpass_kwargs)

        response = system_response.get_response(channel_id=0)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].plot(frequencies, response["gain"](frequencies))
        axs[1].plot(frequencies, response["phase"](frequencies))
        axs[0].set_xlim(0, 1.)
        axs[0].set_title("gain")
        axs[1].set_title("phase")
        fig.suptitle(template_key)
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()
