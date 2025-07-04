#!/bin/bash

STATION=11

#python plotting_scripts/plot_calibration_results.py -d  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets/season2023/station${STATION}/clean/average_ft_combined.pickle \
# -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_05_12_ice/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_05_12_electronic/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_05_12_galactic/station${STATION}/clean/average_ft.pickle

python plotting_scripts/plot_calibration_results.py -d  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets/season2023/station${STATION}/clean/average_ft_combined.pickle \
 -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_06_24_ice/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_06_24_electronic/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_06_24_galactic/station${STATION}/clean/average_ft.pickle
# -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/ice/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/electronic/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/galactic/station${STATION}/clean/average_ft.pickle
