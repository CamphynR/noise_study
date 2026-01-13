#!/bin/bash

STATION=${1:-11}
echo $STATION
SEASON=2023

#python plotting_scripts/plot_calibration_results.py -d  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/flower_average_ft/station_test/average_ft_combined.pickle \
# -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_08_21_ice_flower/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_08_21_electronic_flower/station${STATION}/clean/average_ft.pickle \
#    --filename "test_flower"

python plotting_scripts/plot_calibration_results.py -d  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/average_ft/complete_average_ft_sets_v0.2/season${SEASON}/station${STATION}/clean/average_ft_combined.pickle \
 -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/ice/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/electronic/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_sim_average_ft_set_v0.1/galactic//station${STATION}/clean/average_ft.pickle \
  --save_folder /user/rcamphyn/noise_study/testing_new_calibration
#  --include_impedance_mismatch_correction
# -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/ice/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/electronic/station${STATION}/clean/average_ft.pickle \
#  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/complete_average_ft_sim_sets/galactic/station${STATION}/clean/average_ft.pickle
