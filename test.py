import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--testing")
    parser.add_argument("--bla")
    args = parser.parse_args()

    print(type(args))
    print(args)
    print(type(vars(args)))

    d = dict(a = 3, echo = 5)
    tot = {**vars(args),  **d}
    print(tot)