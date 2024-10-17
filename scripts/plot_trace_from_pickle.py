import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def read_trace_from_pickle(data):
    with open(data, "rb") as f:
        traces = pickle.load(f)
    return np.array(traces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-d", "--data")
    args = parser.parse_args()

    traces = read_trace_from_pickle(args.data)
    fig, ax = plt.subplots()
    ax.plot(traces[0, 0])
    fig.savefig("figures/trace_test.png")