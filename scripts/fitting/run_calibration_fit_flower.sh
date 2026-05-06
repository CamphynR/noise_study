#!/bin/bash

STATION=${1:-11}
echo $STATION

python plotting_scripts/plot_calibration_results_flower.py -d  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/data/flower_average_ft/station${STATION}/average_ft_combined.pickle \
 -s /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_08_29_ice_flower_no_system_response/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_08_29_electronic_flower_no_system_response/station${STATION}/clean/average_ft.pickle \
  /pnfs/iihe/rno-g/store/user/rcamphyn/noise_study/simulations/average_ft/job_2025_08_29_galactic_flower_no_system_response/station${STATION}/clean/average_ft.pickle \
    --filename "test_flower"
