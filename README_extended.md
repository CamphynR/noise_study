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
