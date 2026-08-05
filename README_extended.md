# Extended README

This file complains a (too?) detailed description of the important components in this repo, categorized on workflow steps

## DATA

### Structure
The basic structure of the processing scripts is from top to bottom ``` *.submit < main(_multi).py < main_parser_functions.py ```
Meaning to run the processing one runs a condor_submit file that calls ```main_multi.py ``` or ``` main.py ```. These python scripts are governed by a config file that can be passed as an argument.
The config file chooses hox the data is processed and what parameters are saved. The functions to process the data are stored in ``` main_parser_functions.py ```

### SPECTRA
The data at the lowest level are the root files stored in the T2B mirror
```
/pnfs/iihe/rno-g/data/handcarry
```
a first pass of data processing is performed to only gather spectra of forced trigger data with a maximum trigger rate of 2Hz. No extra claening is done in this step. Spectra are saved per event.
The script used to process data is ``` main_multi.py ```
(note this is not a multiprocessing script in itself it splits the data in batches and you can specify the batch number to process.
The script is used in conjuction with the condor submit script. A typical command to run this first data pass is
```
for st in 11 12 13 21 22 23 24; do condor_submit --append STATION=$st submit/noise_study_spectra.submit; done
```
The settings can be found in ``` configs/config_spectra.json ``` The files are saved as nur files.
The script automatically saves the files in a job folder named after the date the job finishes. The finished sets were placed in
```
/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/spectra/complete_spectra_sets_v0.2
```

##### An important caveat here is that this code uses the NuRadio branch ``` working_branch_noise_study_ruben ``` (the nur files might still contain waveforms stored as spectra but need to check this)

### AVERAGED SPECTRA

The second step of data processing reads in the nur files and applies a channelbandpassfilter and CW filter based on sinewave subtraction. Both of which are specified in the settings file ``` configs/config_average_ft.json```
```
for st in 11 12 13 21 22 23 24; do condor_submit --append STATION=$st submit/noise_study.submit; done
```

This saves the averaged spectra per run as pickle files. Important here is that for this calibration we do not use the standard average but instead we average the squared spectra and take the squareroot at the end.
The reasons for which are specified in the paper.

The resulting files were stored under
```
/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2
```

the averaged spectra per run can be combined into an average over a full season or an average per month. The scripts to do so are stored under
```
scripts/combine_average_ft_squared_batches.py
```
and 
```
scripts/combine_average_ft_squared_monthly.py
```


## Simulation

### Structure

All relevant simulation scripts and settings live in ```sim/thermal_noise``` while some components such as the system response template files live in ``` sim/library```.
```sim/rayleigh``` is legacy that was used to generate noise based on data-driven rayleigh distributions. The code might still work but is outdated.

The simulation consists of noise components [ice, electronic, galactic] **without** the system response template. The system response is applied later during the fitting steps.

### Thermal noise
Thermal noise is first simulated as waveforms and stored as .nur files. Each thermal noise component (ice, electronic, galactic) is simulated seperately and stored in a seperate .nur file.
The master script to run a simulation is
```
sim/thermal_noise/generate_thermal_noise_traces.py
```
this code takes a config file as an argument. The default config file is stored under ```sim/thermal_noise/configs/config_sim.json``` .
The script to submit simulation jobs to the T2B cluster is
```
sim/thermal_noise/sim.submit
```

The generated thermal noise files were saved to
```
/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/thermal_noise_traces/default
```

#### sim config file structure
Some settings in the sim's config file warrant some explanation.

``` digitizer ``` : define which digitizer to use to run the simulation. Only has an effect on the sampling rate (for now)

``` use_templates ``` when set to True the sim will not run a simulation for each channel in a station but will instead use a limited set of templates. e.g. one template for VPols at depth -100m is simulated and copied to all VPols at -100m

``` template_channels ``` the channels to use as templates. To be defined per noise component. e.g. for electronic noise the only templates are the deep and surface components. Hence one deep channel and one surface channel are chosen as templates.

``` template_mapping ``` define which channels should use which template. Again per noise component. e.g. for electronic noise all deep channels should use the deep template. So channel 0 is mapped to all deep channels

``` channel_types ``` Define groups of channels per type of antenna

``` antenna_models ``` Define which antenna model to use for which channel group

These last two options could probably be merged but it was more readable in the code to keep it like this.

### Average spectra

The simulated thermal noise traces are processed in the same way as the data spectra. The only difference is that the noise components are processed seperately. This is done automatically by the main scripts by detecting whether it's input is simulation or data. It does this with a sim=True or False tag in the config file saved together with the spectra files.

Since there are not as many simulations as data the simulation processing uses ``` main.py ``` . The submit file is
```
submit/noise_study_sim.submit
```
with the config file
```
configs/config_average_ft_sim.json
```
the bandpass filter for the simulation is not included in this step since it is already included later when applying the system response template.

The final processed files are stored in
```
/pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/default
```
The averages are taken per noise component due to the way we fit later. The fitting also requires the cross products between the noise components. To generate these run the script
```
scripts/calculate_cross_products_sim_components.py
```
The file and save paths are defined in the script itself.


## Fitting

### Structure

The master fitting script is
```
scripts/fitting/run_calibration_fit.py
```
the fitting script makes use of the class ``` fitting/spectrumFitter ```. This class contains all different possible fit functions. It also includes optional weights to scale each noise component individually for testing purposes. It's init also accepts a system response, which is an instance of ```modules/systemResponseTimeDomainIncorporator.py``` and uses the class to apply the system response templates.

### Master script settings

The master script's argument parser contains several settings such as which data or which simulation paths to use. If not specified the script uses the paths specified above. The first section of the code contains settings for the fitting.

The most important one is the mode selection on line ```154```. This allows you to choose the actual function that is fitted e.g. ```constant``` only fits the gain factor G. The default is ```system_response_weight``` which also includes a slope weight and uses three parameters: gain, slope and f0.

The ```parameter_fixed``` setting dictates which fitting parameters are kept fixed. If this list contains multiples settings the fitter will interpret these as fit steps. e.g. [[False, True, True], [True, False, True]] will first fit the gain while keepng the other parameters fixed and the fit the slope while keeping the gain and f0 fixed. Which index corresponds to which parameter is defined by the mode in spectrumFitter.

``` cable_length ``` on line ```261``` defines the length of the coax cable that is removed from the surface response and is given as an option to ```spectrumFitter```

Also of not is that in line ```320``` the pulser response of the pulser used to measure the system response templates is defined as a weight to be divided out of the system response templates. This is done by passing a weight option to ```systemResponseTimeDomainIncorporator```

Line ```340``` contains the actual fitting loop. The code loops over all system response templates, performs the fit and saves the goodness of fit values. Afterwards the templates with the best (lowest) goodness of fit scores are saved. Fit results by default are stored in

```absolute_amplitude_results/seasonXXXX/stationXX/default```

if ```args.fname_appendix``` is defined ```default``` will be replaced by ```args.fname_appendix```. The results for each template are stored, while the compiled results with the best template fit are stored with the ```best_fit``` appendix. The plot points used in the calibration plots are also saved here as a pickle file.

All results are also plotted and saved under
```
figures/absolute_ampl_calibration/spectra_fit_seasonXXXX_stXX_all_template_fit.pdf
```
note that if defined ```args.fname_appenidx``` is included in this name.

The best fit results are plotted as 
```
figures/absolute_ampl_calibration/spectra_fit_seasonXXXX_stXX_best_template_fit.pdf
```

The script
```
scripts/fitting/run_calibration_fit_per_run.py
```
is a convenience script that is used to run a calibration per run of data. It skips the plotting to improve run time and only fits the best template found by the full season fit.

## Paper plots
(Ignore ```plotting_scripts/paper``` that is outdated)

Figure 4 (figure on effective ice temperature in function of incidence angle) is generated in ```ice_integrator/ice_temperature.ipynb```

Figure 6 (electronic noise temperature) is generated with ```plotting_scripts/electronic_noise/plot_noise_temperature.py```

Figure 7 (compilation of representative calibration results) is generated with ```plotting_scripts/plot_data_vs_calibrated_sim.py```

Figure 8 (overview of all gains) is generated with ```plotting_scripts/calibration_summary_plots/plot_gain_overview.py```

Figure 9 (calibration over time) is generated with ```plotting_scripts/calibration_summary_plots/plot_calibration_over_time_all.py```

Figure 10 (forced trigger Vrms validation)  is generated with ```plotting_scripts/validation/plot_vrms_distribution_calibrated.py```


## Extra notes

Most systematics were investigated using the plotting scripts under
```
plotting_scripts/systematics
```
