import math
import numpy as np
import logging
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

from radiotools import helper as hp


def check_solution(ray_tracer, x2,x1,ice,freq, fmax = 1*units.GHz):
    attn_isnan = False
    direct_refrac_sol = False
    type_sol = 0
    attn = np.zeros(len(freq))
    ray_tracer.reset_solutions()
    ray_tracer.set_start_and_end_point(x1, x2)
    ray_tracer.find_solutions()
    iS_r = -1
    if(ray_tracer.has_solution()):
        for iS in range(ray_tracer.get_number_of_solutions()):
            if(ray_tracer.get_solution_type(iS) >2):
                continue
            direct_refrac_sol = True
            attn = ray_tracer.get_attenuation(iS, freq , fmax)
            iS_r = iS
            if(math.isnan(attn[0])):
                attn_isnan = True
                attn = np.zeros(len(freq))
            break
    return direct_refrac_sol,attn_isnan,attn,iS_r


def Add_attenuation_piece_of_ice(theta,d_theta ,radius,step,z_antenna,freq,ice,fmax, ray_tracer):
        print("starting attenuation")
        z = z_antenna+np.cos(theta/units.rad)*radius
        x2 = np.array([np.sin(theta/units.rad)*radius, 0., z]) #efield
        x1 = np.array([0., 0., z_antenna])  # antenna
        direct_refrac_sol,attn_isnan,attn,iS_r = check_solution(ray_tracer, x2,x1,ice,freq,fmax)
        radius_new = radius
        while(direct_refrac_sol and attn_isnan and (radius_new<radius+(step)/2)):
            radius_new+=(step)/20
            z_new = z_antenna+np.cos(theta/units.rad)*radius_new
            x2_new = np.array([np.sin(theta/units.rad)*radius_new, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,iS_r = check_solution(ray_tracer, x2_new,x1,ice,freq,fmax)
        radius_new = radius
        while(direct_refrac_sol and attn_isnan and (radius_new>radius-(step)/2)):
            radius_new-=(step)/20
            z_new = z_antenna+np.cos(theta/units.rad)*radius_new
            x2_new = np.array([np.sin(theta/units.rad)*radius_new, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,iS_r = check_solution(ray_tracer, x2_new,x1,ice,freq,fmax)
        theta_new = theta
        while(direct_refrac_sol==False and theta_new<(theta+d_theta/2)):
            theta_new+=d_theta/50
            theta_new = round(theta_new/units.degree,1)*units.degree
            z_new = z_antenna+np.cos(theta_new/units.rad)*radius
            x2_new = np.array([np.sin(theta_new/units.rad)*radius, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,iS_r = check_solution(ray_tracer, x2_new,x1,ice,freq,fmax)
        theta_new = theta
        while(direct_refrac_sol==False and theta_new>(theta-d_theta/2)):
            theta_new-=d_theta/50
            theta_new = round(theta_new/units.degree,1)*units.degree
            z_new = z_antenna+np.cos(theta_new/units.rad)*radius
            x2_new = np.array([np.sin(theta_new/units.rad)*radius, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,iS_r = check_solution(ray_tracer, x2_new,x1,ice,freq,fmax)
        zenith = -1
        sol_type = 0
        if(direct_refrac_sol):
            receive_vector = ray_tracer.get_receive_vector(iS_r)
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            sol_type = ray_tracer.get_solution_type(iS_r)
        #check annt
        for i in range(len(attn)):
            if(math.isnan(attn[i])):
                attn[i] = 0
        print("finished one attenuation")
        return attn,zenith,sol_type



medium = medium.greenland_simple()

ray_tracer = ray.ray_tracing(medium=medium,
                            attenuation_model="GL3",
                            log_level=logging.DEBUG)

radius = 400*units.m
step = 50 * units.m
theta = 140 *units.degree
d_theta = 5*units.degree
z_antenna = -100*units.m

x2 = [radius * np.sin(theta), 0., radius * np.cos(theta) - z_antenna]
x1 = [0., 0., z_antenna]

sampling_rate = 3.2*units.GHz
freq = np.fft.rfftfreq(2048, d=1./sampling_rate)
print(freq)
attn, zenith, sol_type = Add_attenuation_piece_of_ice(theta, d_theta, radius, step, z_antenna, freq, ice=medium,
                             fmax=sampling_rate/2, ray_tracer=ray_tracer)

print(attn)
