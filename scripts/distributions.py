"""
File should contain code to generate bins to plot a distribution, given a pickle of variables. These bins should be saved in a seperate pickle.
"""
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d")
    args = parser.parse_args


