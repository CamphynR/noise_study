import json
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units

from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator


if __name__ == "__main__":
    debug = True

    sampling_rate_v2 = 3.2 * units.GHz
    sampling_rate_v3 = 2.4 * units.GHz

    template_filepaths = ["sim/library/deep_templates_combined.json",
                          "sim/library/v2_v3_surface_impulse_responses.json"]

    template_keys = []
    for filepath in template_filepaths:
        with open(filepath, "r") as file:
            file_keys = list(json.load(file).keys())
        file_keys.remove("time")
        template_keys += file_keys
    template_keys += ["surface_query"]
    

    if debug:
        print(f"Found keys: {template_keys}")

    
    template_dir = "/user/rcamphyn/noise_study/sim/library/templates/"
    
    for template_key in template_keys:
        if debug:
            print(template_key)

        # name starting without a v also pertain to radiant v2's
        if template_key.startswith("v3"):
            sampling_rate = sampling_rate_v3
        else:
            sampling_rate = sampling_rate_v2

        response_helper = systemResponseTimeDomainIncorporator()
        response_helper.begin(det=0,
                              response_path=template_filepaths,
                              overwrite_key=template_key)

        template_filename = template_key + ".json"
        template_filename = template_dir + template_filename

        response_helper.save_response(template_filename,
                                      sampling_rate = sampling_rate,
                                      channel_id=0)

        if debug:
            with open(template_filename, "r") as template_file:
                template = json.load(template_file)
            frequencies = template["frequencies"]
            gain = template["gain"]
            phase = template["phase"]

            plt.style.use("retro")
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].plot(frequencies, gain)
            axs[1].plot(frequencies, phase)

            for ax in axs:
                ax.set_xlabel("frequencies / GHz")
            axs[0].set_ylabel("Gain amplitude")
            axs[1].set_ylabel("Phase")
            
            fig_dir = "/user/rcamphyn/noise_study/figures/debug/saving_templates/"
            fig.savefig(fig_dir + template_key + ".png")
            plt.clf(fig)



