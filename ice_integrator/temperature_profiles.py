import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



class temperature_profile_grip:
    def __init__(self, debug=False):
        data = np.loadtxt("griptemp.txt", skiprows=39)
        self.depth_values = -data[:,0]
        self.temperature_values = data[:,1]
        self.profile = interp1d(self.depth_values,
                                self.temperature_values,
                                bounds_error=False,
                                fill_value=(self.temperature_values[-1],self.temperature_values[0]))
        if debug:
            plt.plot(self.profile(np.linspace(-1,-3000,10)), np.linspace(-1,-3000,10))

    def attenuation_from_temperature(self, T, debug=False):
        # temperature, attenuation, using PlotDigitizer from http://dx.doi.org/10.3189/2015JoG15J057
        # -35, 1304.5092245697026
        # -29.980988593155892: 1073.7280752306615
        # -25.076045627376427: 883.7744937517206
        # -20.019011406844108: 716.6127862412111
        # -15: 589.8365861768901
        # -10.019011406844108: 481.8666050152245
        # -5: 396.61942788606507
        T0 = -5
        dT = -35 - (-5)
        dAtt = np.log10(397) - np.log10(1304)
        Att = -dAtt/dT*(T-T0) + np.log10(397)
        if debug:
            plt.plot(np.linspace(-5, -35, 10), [attenuation(x) for x in np.linspace(-5, -35, 10)] )
            plt.xlim(-35, -5)
            plt.semilogy()
            plt.ylim(1e2, 1e4)
        return 10**Att

    def attenuation_from_depth(self, depths):
        T = self.profile(depths)
        return self.attenuation_from_temperature(T)

