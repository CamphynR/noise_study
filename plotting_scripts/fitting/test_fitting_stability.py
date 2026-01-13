import argparse
import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter

from fitting.spectrumFitter import spectrumFitter
from modules.systemResponseTimeDomainIncorporator import systemResponseTimeDomainIncorporator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # SETTINGS
    plt.style.use("retro")
    system_response_path = ["sim/library/v2_v3_deep_impulse_responses_for_comparison.json",
                            "sim/library/v2_v3_surface_impulse_responses.json"]
    season = 2023
    if season > 2023:
        digitizer_version = "digitizer_v3"
        system_response_keys = {"deep" : "v3_ch4_62dB",
                                "helper" : "v3_ch4_62dB",
                                "surface" : "v3_ch14"}
    else:
        digitizer_version = "digitizer_v2"
        system_response_keys = {"deep" : "v2_ch2",
                                "helper" : "v2_ch9",
                                "surface" : "v3_ch14"}

    system_response_bandpass_kwargs = dict(
            passband=[0.1, 0.7],
            filter_type="butter",
            order=10
            )

    station_id = 11
    channel_id = 19


    # DATA
    data_path = f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season{season}/station{station_id}/clean/average_ft_combined.pickle"

    sim_paths = [f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1_no_system_response/{digitizer_version}/ice/station{station_id}/clean/average_ft.pickle",
                 f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1_no_system_response/{digitizer_version}/electronic//station{station_id}/clean/average_ft.pickle",
                f"/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1_no_system_response/{digitizer_version}/galactic/station{station_id}/clean/average_ft.pickle"
                 ]


    # SYSTEM RESPONSE
    system_response = systemResponseTimeDomainIncorporator()
    system_response.begin(
            det=0,
            response_path=system_response_path,
            overwrite_key=system_response_keys,
            bandpass_kwargs=system_response_bandpass_kwargs)

    if args.test:
        test_response = system_response.get_response(channel_id=0)
        test_freqs = np.arange(0., 1., 0.01)
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].plot(test_freqs, test_response["gain"](test_freqs))
        axs[0].set_ylabel("gain")
        axs[1].plot(test_freqs, test_response["phase"](test_freqs))
        axs[1].set_ylabel("phase")
        for ax in axs:
            ax.set_xlabel("freqs / GHz")
        fig.savefig("test_fitting_stability_response_test.png")
        plt.close(fig)



    
    # FITTER
    fitter = spectrumFitter(data_path, sim_paths,
                            system_response=system_response)


    # FITTING SETTINGS


    # different modes
    nr_steps = 30


    mode_gain_el_ampl_split = {"fit_function" : "electronic_temp",
            "steps" : {}}
    for i in range(1, nr_steps, 2):
        mode_gain_el_ampl_split["steps"].update(
                {
                    f"step {i}" : {"gain" : False, "el_ampl" : True, "el_cst" : True, "f0" : True},
                    f"step {i+1}" : {"gain" : True, "el_ampl" : False, "el_cst" : False, "f0" : True}
                    })
                        


    mode_no_el_cst = {"fit_function" : "electronic_temp",
            "steps" : {}}
    for i in range(1, nr_steps, 2):
        mode_no_el_cst["steps"].update(
                {
                    f"step {i}" : {"gain" : False, "el_ampl" : True, "el_cst" : True, "f0" : True},
                    f"step {i+1}" : {"gain" : True, "el_ampl" : False, "el_cst" : True, "f0" : True}
                    })


    mode_all_variables = {"fit_function" : "electronic_temp",
            "steps" : {}}
    for i in range(1, nr_steps, 2):
        mode_all_variables["steps"].update(
                {
                    f"step {i}" : {"gain" : False, "el_ampl" : True, "el_cst" : True, "f0" : True},
                    f"step {i+1}" : {"gain" : False, "el_ampl" : False, "el_cst" : False, "f0" : True}
                    })


    mode_only_gain = {"fit_function" : "electronic_temp",
            "steps" : {}}
    for i in range(1, nr_steps):
        mode_only_gain["steps"].update(
                {
                    f"step {i}" : {"gain" : False, "el_ampl" : True, "el_cst" : True, "f0" : True}
                    })




    # PLOTTING
    mode = mode_no_el_cst

    results_per_step = []
    mode_up_to_step = {}
    mode_up_to_step["fit_function"] = mode["fit_function"]
    mode_up_to_step["steps"] = {}
    for step in range(1, nr_steps):
        mode_up_to_step["steps"].update(list(mode["steps"].items())[:step])
        result = fitter.get_fit_gain(mode=mode_up_to_step, choose_channels=[channel_id])
        results_per_step.append(result)

    
    # PLOTTING
    
    fig, (ax_gain, ax_el_ampl, ax_el_cst) = plt.subplots(3, 1, sharex=True, figsize=(20, 20))
    gains_per_step = [result[0]["gain"].value for result in results_per_step]
    el_ampl_per_step = [result[0]["el_ampl"].value for result in results_per_step]
    el_cst_per_step = [result[0]["el_cst"].value for result in results_per_step]

    ax_gain.plot(range(1, nr_steps), gains_per_step)
    ax_gain.set_ylabel("gain")
    ax_el_ampl.plot(range(1, nr_steps), el_ampl_per_step)
    ax_el_ampl.set_ylabel("el_ampl")
    ax_el_cst.plot(range(1, nr_steps), el_cst_per_step)
    ax_el_cst.set_ylabel("el_cst")


    ax_el_cst.set_xticks(range(1, nr_steps), labels=range(1, nr_steps),
                  rotation=90)
    ax_el_cst.set_xlabel("steps")

    fig.suptitle("G - A,B alternating")
    fig.savefig("step")
