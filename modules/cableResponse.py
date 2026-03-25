import numpy as np
from scipy.interpolate import interp1d

from utilities.utility_functions import convert_db_to_amplitude




class cableResponse():
    def __init__(self, length):
        # point taken from plot felix
        self.frequencies = [0.05, 0.15, 0.22, 0.45, 0.9]
        # this describes attenuation so the db is negative
        db_per_100m = [-2.95, -4.92, -6.23, -8.86, -12.8]
        gain_db = [db * (length/100) for db in db_per_100m]
        gain_amplitude = [convert_db_to_amplitude(db) for db in gain_db]
        # fill with 1's to indicate no change
        self.gain_amplitude = interp1d(self.frequencies, gain_amplitude, bounds_error=False, fill_value=1.)

        return

    def get_gain(self):
        return self.gain_amplitude







if __name__ == "__main__":
    import matplotlib.pyplot as plt

    freqs = np.linspace(0, 1., 1000)

    cable_response = cableResponse(length=11)


    amp = cable_response.get_gain()(freqs)


    plt.style.use("retro")
    plt.plot(freqs, amp)
    plt.xlabel("frequency / GHz")
    plt.ylabel("response / amplitude")
    plt.savefig("figures/test_cable_response_module.png")
