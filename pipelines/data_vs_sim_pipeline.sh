#!/bin/bash

STATION=12

source ~/envs/mattak_env.sh

# generate file saves job directory to SIM_JOB_PATH environment
echo "Starting simulation"
sim_path=$(nice -1 python sim/thermal_noise/generate_thermal_noise_traces.py -s $STATION)

echo $sim_path

echo "Starting main.py"
main_path=$(nice -1 python main.py -d "$sim_path" -s "$STATION")
IFS=$'\n'
i=0
for sim in $main_path; do
    eval sim_path_${i}="$sim"
    i=$((i+1))
    done

echo "Starting plot"
nice -1 python plotting_scripts/plot_average_ft.py \
-d "/home/ruben/Documents/data/noise_study/data/average_ft/complete_runs/season2023/station$STATION/average_ft_combined.pickle" \
-s "$sim_path_0"/station$STATION/clean/average_ft.pickle \
   "$sim_path_1"/station$STATION/clean/average_ft.pickle \
   "$sim_path_2"/station$STATION/clean/average_ft.pickle \
   "$sim_path_3"/station$STATION/clean/average_ft.pickle
