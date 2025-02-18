# Ice temperature seen by antennas
## Requirements
* git@github.com:RNO-G/antenna-positioning.git
* git@github.com:nu-radio/RadioPropa.git
  not sure everything needed for the polynomial ice model is in the main branch by now - it should since Bob is defending ;). I likely used the `ice_model/exponential_polynomial` branch at the time. 
## What this is about
* I use radiopropa to shoot out rays under different theta angles from an antenna and propagate them outward to -3000 m (the bottom) and 10 m height (above ice) as boundary layers. The trajectory of the in-ice ray (may be reflected) is used to integrate the temperature profile along the trajectory with emission and absorption. The `ice_integrator.py` does this. Note: The calculation could be much more convenient, at the moment temporary files holding the trajectory are written and read in. This can be avoided, see 1) in the things to improve section.
* The GRIP temperature profile is used `temperature_profiles.py` to get the temperature as a function of depth, and also the attenuation as a function of temperature and hence depth. See also `ice_temperature_profile_plots.ipynb` for some plots of the implemented profile.
* Using the radiopropa trajectories and the temperature and attenuation as a function of depth, the noise temperature seen under a specific theta angle is then calculated along the trajectory.
* To obtain the effective temperature integrated over the entire antenna the theta-dependent vector effective length is folded with the theta-dependent temperature seen. This is done in `ice_temperature.ipynb`. The most relevant function to call is probably `effective_temperature_for_channel(cc, depth=dd)` which does exactly that.
## Things to improve / continue
* 1) The radiopropa part writes out a temporary file and reads the data from that. It would be more straightforward to do something in the lines of:
  ```
      source = radiopropa.Source()
      source.add(radiopropa.SourcePosition(radiopropa.Vector3d(0, 0, z)))
      dir_z = np.cos(phi * radiopropa.deg)
      dir_x = np.sin(phi * radiopropa.deg)
  
      source.add(radiopropa.SourceDirection(radiopropa.Vector3d(dir_x, 0 , dir_z)))
  
      #print('Ray Direction {} deg = ({}, 0, {})'.format(phi, x, z))
      sim.setShowProgress(True)
      ray = source.getCandidate()
      sim.run(ray, True)
  ```
  The relevant part is the last two lines: A ray is defined and the sim is run for that ray. Afterwards the ray can be used to get the trace.
* 2) The relevant integration functions should be moved out from the jupyter notebook to a script/class.
* 3) Be aware that the calculated temperature is the external noise temperature AT THE ANTENNA. I.e. one would need to treat this temperature similar to the Galactic noise adder (but it is constant over daytime and hence simpler) to get to the total noise of the system. The thermal noise of the system is not taken into acount with this temperature and should be added by the generic noise adder. Note: the noise temperature plugged into the generic noise adder would tend to be lower, since it would then only simulate the noise of the system, but not the noise temperature seen by the antenna any more.