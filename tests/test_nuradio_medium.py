
from astropy.time import Time
from NuRadioReco.detector.detector import Detector
from NuRadioMC.utilities import medium


detector = Detector(source="rnog_mongo", select_stations=23)
det_time = Time("2022-08-01")
detector.update(det_time)
ice = medium.greenland_simple()
print(ice)
