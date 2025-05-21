# RNO-G absolute system response calibration

This repository contains the code to construct, develop and test the thermal noise system response calibration.
The calibration has a wiki page, which can be found here: https://radio.uchicago.edu/wiki/index.php/Thermal_noise_for_detector_response_calibration 





## Notes to self:
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
