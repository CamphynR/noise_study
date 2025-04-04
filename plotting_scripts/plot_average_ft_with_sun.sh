#!/bin/bash

python plotting_scripts/plot_ft_sun.py -d  ~/Documents/data/noise_study/data/average_ft/job_2025_02_21_test/station23/clean/average_ft.pickle \
                                       -s ~/Documents/data/noise_study/simulations/average_ft/job_2025_03_05_ice/station23/clean/average_ft.pickle \
                                          ~/Documents/data/noise_study/simulations/average_ft/job_2025_03_05_electronic/station23/clean/average_ft.pickle \
                                          ~/Documents/data/noise_study/simulations/average_ft/job_2025_03_01_galactic/station23/clean/average_ft.pickle \
                                          ~/Documents/data/noise_study/simulations/average_ft/job_2025_03_05_galactic_sun/station23/clean/average_ft.pickle \
                                          ~/Documents/data/noise_study/simulations/average_ft/job_2025_03_05_sum/station23/clean/average_ft.pickle
