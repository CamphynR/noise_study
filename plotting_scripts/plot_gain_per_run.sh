#!/bin/bash

STATION=22
SEASON=2023

python plotting_scripts/plot_gain_per_run.py --data  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.1/season${SEASON}/station${STATION}/clean/average_ft_run* \
    --sims /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/ice/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/electronic/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/galactic/station${STATION}/clean/average_ft.pickle
