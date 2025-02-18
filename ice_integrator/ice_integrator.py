import matplotlib.pyplot as plt
import os
import h5py
import radiopropa
import numpy as np
import sys
from NuRadioMC.utilities import medium
from AntPosCal.ice_model import icemodel

def get_angles_rhos(file):
    f = h5py.File(file)
    d = f['Trajectory3D']

    # only take the reflection
    d = d[d["SN"] == d["SN"][0]]
    X = d['X']
    Z = d['Z']
    # starting angle from first step
    dX = X[1] - X[0]
    dZ = Z[1] - Z[0]
    theta0 = 180*abs(np.arctan(dX/dZ))/np.pi
    distance = d['D']
    power = (d['Ax']**2 + d['Ay']**2 + d['Az']**2)
    #print(d["SN"])
    return theta0, X, Z, distance, power

def get_trace_file(theta, z_antenna):
        # use fifth order polynomial ice model here, I checked that index of refraction is 1. at 0+epsilon and 1.27 at 0-epsilon
        icename = "greenland_poly5"
        ice = icemodel.greenland_poly5()
        iceModelScalar = ice.get_ice_model_radiopropa().get_scalar_field()

        # simulation setup
        sim = radiopropa.ModuleList()
        sim.add(radiopropa.PropagationCK(iceModelScalar, 1E-8, .001, 1.))

        # add a discontinuity
        firnLayer = radiopropa.Discontinuity(radiopropa.Plane(radiopropa.Vector3d(0,0,-0.001), radiopropa.Vector3d(0,0,1)), iceModelScalar.getValue(radiopropa.Vector3d(0,0,-0.001)) , 1.)

        # add a reflective layer at the surface
        reflective = radiopropa.ReflectiveLayer(radiopropa.Plane(radiopropa.Vector3d(0,0,-0.001), radiopropa.Vector3d(0,0,1)),1)
        sim.add(firnLayer)

        # Observer to stop simulation at z=0m and z=-300m
        #obs = radiopropa.Observer()
        #obsz = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0,0,0), radiopropa.Vector3d(0,0,1)))
        #obs.add(obsz)
        #obs.setDeactivateOnDetection(False)
        #sim.add(obs)

        obs2 = radiopropa.Observer()
        obsz2 = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0,0,-3000), radiopropa.Vector3d(0,0,1)))
        obs2.add(obsz2)
        obs2.setDeactivateOnDetection(True)
        sim.add(obs2)

        obs3 = radiopropa.Observer()
        obsz3 = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0,0,10), radiopropa.Vector3d(0,0,1)))
        obs3.add(obsz3)
        obs3.setDeactivateOnDetection(True)
        sim.add(obs3)


        # Output
        tempfile = '__output_tempfile.h5'
        if os.path.isfile(tempfile):
            os.remove(tempfile)
        output = radiopropa.HDF5Output(tempfile, radiopropa.Output.Trajectory3D)
        output.setLengthScale(radiopropa.meter)
        #output.enable(radiopropa.Output.CurrentAmplitudeColumn)
        output.enable(radiopropa.Output.SerialNumberColumn)
        sim.add(output)

        # Source
        source = radiopropa.Source()
        source.add(radiopropa.SourcePosition(radiopropa.Vector3d(0, 0, z_antenna)))
        source.add(radiopropa.SourceAmplitude(1))
        source.add(radiopropa.SourceFrequency(403E6))
        z = np.cos(theta * radiopropa.deg)
        x = np.sin(theta * radiopropa.deg)
        source.add(radiopropa.SourceDirection(radiopropa.Vector3d(x, 0 , z)))
        sim.setShowProgress(False)
        sim.run(source, 1)
        return tempfile

def get_trace(theta, z_antenna):
    file = get_trace_file(theta, z_antenna)
    theta, X, Z, D, A = get_angles_rhos(file)
    return X, Z, D, A

if __name__ == "__main__":
    for theta in np.linspace(0,60,10):
        X, Z, distance, power = get_trace(theta, z_antenna=-100)
        plt.plot(distance, power, label=f"{theta}", alpha=0.5)
    plt.legend()
    #plt.xlim(0, 6000)
    plt.xlabel("distance [m]")
    plt.ylabel("power")
    plt.savefig("plot.pdf")
