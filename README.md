# RNO-G absolute system response calibration

This repository contains the code to construct, develop and test the thermal noise system response calibration.
The calibration has a wiki page, which can be found here: https://radio.uchicago.edu/wiki/index.php/Thermal_noise_for_detector_response_calibration 

It also includes code used to construct data-driven rayleigh noise

# Disclaimer

### Running the code

Be careful when running the code out of the box. Files are saved to the /tmp folder before being copied to the given save_dir (due to the HTCondor cluster on which this code is used)
So your /tmp folder could max out which is not good!

### Storage

This repo contains a full copy of the working environment in which this calibration was produced, including .png, .pdf and .json files. Be careful when cloning the repo as it might be quite big.

## Calibration

The calibration aims to compare forced tigger data with simulations in order to find an absolute system gain factor.
The code for processing data is contained in main.py, main_multi.py and main_parser_functions.py
main_multi.py is used for processing data, main.py is used for processing simulations
The code uses a config file in which to specify what variables to calculate, what cleaning to apply, data ranges and voltage calibration
To calculate spectra, the config file takes the variable "average_ft"
The configs are placed under the configs folder, with config_all_options.json acting as a template.
The config is given to the code as python main_multi.py --config configs/config_all_options.json

The code for simulating thermal noise is under sim/thermal_noise/generate_thermal_noise_traces.py
The simulations also use a config file, contained under sim/thermal_noise
The folder sim/library contains all files used by the simulation software
The simulations save thermal noise traces. To obtain spectra the main.py function is used.

The final fitting is done by the spectrumFitter module under fitting/spectrumFitter.py


## Note on station temperatures
This folder normally includes csv files of temperature measurments but the files are too big for a github repo.
The files are generated using Matthew's MonitoringDataAcces, which can be found on the RNO-G repo


## CLEANING
use as event selection:
    type: "PHYSICS"
    trigger rate < 2 Hz (higher trigger rates are probably windy/human activity periods)

### BROKEN RUNS:
The dataset contains several broken/incomplete runs. To filter these a python program sweeps over all runs and stores the run_nr of the ones that fail to load. These numbers are stored in broken_runs/stationX.pickle. The main.py program uses these broken run pickle files to ignore the broken runs. An example of a broken run is station 24, run 363, event 105 is broken (200 header entries, but only 100 waveforms)

### Calibration:
The calibration is available but for season 22 data we choose to use a linear calibration. The reason is that the amount of bias scans in the chosen time frame is limited.

### Potential artifacts:
    Have a look at ringing, which can be caused be the deconvolution of the detector out of the trace

What to do with G = 0? this represents e.g. a bandpass filter.
This information is lost so what value should be used as placeholder or this should be deleted?
For the purpose of thermal noise from ice I assume these can be set to 0 since the signal here is purely electronics noise? -> The solution was to use a bandpass filter which anyways focuses on "normal" response values.
