
import matplotlib.pyplot as plt
import numpy as np




class pulserResponse:
    def __init__(self):
        # in 1/GHz
        self.slope = -0.352
        # in GHz
        self.f0 = 0.1

    def frequency_response(self, frequencies):
        return 1.0 + self.slope * (frequencies - self.f0)

    def __call__(self, frequencies):
        return self.frequency_response(frequencies)




if __name__ == "__main__":
    pulser = pulserResponse()
    freqs = np.arange(0, 1., 0.001)
    response = pulser(freqs)

    fig, ax = plt.subplots()
    ax.plot(freqs, response)
    ax.set_ylim(0.55, 1.45)
    fig.savefig("test")
