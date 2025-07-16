#!/bin/bash

STATION=11

python plotting_scripts/plot_gain_per_run.py --data  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/job_2025_04_16/station${STATION}/clean/average_ft_batch* \
    --sims /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_04_16_ice/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_04_16_electronic/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_04_16_galactic/station${STATION}/clean/average_ft.pickle 
