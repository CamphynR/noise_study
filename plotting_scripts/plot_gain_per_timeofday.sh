#!/bin/bash

STATION=11

python plotting_scripts/plot_gain_per_timeofday.py --data  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets/season2023/station${STATION}/clean/average_ft_run* \
    --sims /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/ice/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets//electronic/station${STATION}/clean/average_ft.pickle \
    /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/galactic/station${STATION}/clean/average_ft.pickle \
    --channels 0 19 --vector_image --test
