def plot_ft(channel, ax, fs = 3.2e9, label = None):
    from NuRadioReco.utilities import fft

    trace = channel.get_trace()
    ft = fft.time2freq(trace, fs)

    ax.plot(np.abs(ft), label = label)
    ax.legend(loc = "best")