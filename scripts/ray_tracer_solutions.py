
import logging
import numpy as np
import matplotlib.pyplot as plt
from NuRadioMC.SignalProp import analyticraytracing as ray
import NuRadioMC.utilities.medium as medium
from NuRadioReco.utilities import units

z_antenna = -100 * units.m
r0 = 20 * units.cm
R = 2000 * units.m
n_r = 50
radial_d = np.linspace(float(r0/units.m),float(R/units.m), n_r)*units.m
d_theta = 5 * units.degree
thetas = np.arange(d_theta/2,180*units.degree,d_theta)

ice_model = medium.greenland_simple()
attenuation_model = "GL1"

sampling_rate = 3.2 * units.GHz
freqs = np.fft.rfftfreq(2048, d=1./sampling_rate)

ray_tracer = ray.ray_tracing(medium=ice_model,
                             attenuation_model=attenuation_model,
                             log_level=logging.ERROR)

x2_list = []
solution_type = []
attenuation = []
for theta in thetas:
    print(theta)
    for radius in radial_d:
        z = z_antenna+np.cos(theta/units.rad)*radius
        x2 = np.array([np.sin(theta/units.rad)*radius, 0., z]) #efield
        x2_list.append(x2)
        x1 = np.array([0., 0., z_antenna])  # antenna
        
        ray_tracer.reset_solutions()
        ray_tracer.set_start_and_end_point(x1, x2)
        ray_tracer.find_solutions()
        if ray_tracer.has_solution():
#            for iS in range(ray_tracer.get_number_of_solutions()):
#                if ray_tracer.get_solution_type(iS) == 2:
#                    attn = ray_tracer.get_attenuation(iS, freqs)
#                    continue
#                else:
#                    attn = 0
            attn = ray_tracer.get_attenuation(0, freqs)
            attenuation.append(np.mean(attn))
        else:
            attenuation.append(0)
        solution_type.append(ray_tracer.get_number_of_solutions())

x2_list = np.array(x2_list)
fig, ax = plt.subplots()
def type_to_c(type):
    if type == 3:
        return "purple"
    elif type == 2:
        return "green"
    elif type == 1:
        return "blue"
    else:
        return "red"
colors = [type_to_c(nr_sols) for nr_sols in solution_type]
cax = ax.scatter(x2_list[:, 0], x2_list[:, -1], c=attenuation, cmap="inferno")
cbar = fig.colorbar(cax)
ax.scatter(x1[0], x1[-1], color = "green", label = "antenna")
ax.legend()
ax.set_xlabel("horizontal coordinate / m")
ax.set_ylabel("vertical coordinate / m")
ax.set_ylim(-1*R, 0)
fig.savefig("testing_ray_solutions")
