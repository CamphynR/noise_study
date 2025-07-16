from astropy.coordinates import SkyCoord
from astropy.time import Time
import datetime
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from pygdsm import GlobalSkyModel, GSMObserver

from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.modules.channelGalacticNoiseAdder import get_local_coordinates
from NuRadioReco.utilities import units















if __name__ == "__main__":
    # constants
    station_id=11
    time="2023-08-01T00:00:00"
    elevation = 3216 * units.m
    freq = 200
    dec_cut = -17.433333333333337

    det = Detector(select_stations=station_id)
    det.update(Time(time))
    site_lat, site_lon = det.get_site_coordinates(station_id)




    gsm = GlobalSkyModel(freq_unit="MHz")
    sky_map = gsm.generate(freq)
    # by default pygdsm nside=512
#    local_coord = get_local_coordinates((site_lat, site_lon), time, n_side=512)
#    mask = local_coord.alt.rad < 0
#    mask = mask > 0
#        
#    sky_map[mask] = 0

    plt.style.use("retro")
#    hp.mollview(np.log2(sky_map), coord=["G", "C"], title="Galactic noise temperature as seen from RNO-G's location", rot=(180, 0, 0),
#                hold=True,
#                unit=r"log($T_{Ant}$)",
#                )
    spacing = 60
    spacingfrac = spacing/360
    cmap = plt.get_cmap()
    fig, ax = plt.subplots()
    hp.projview(np.log2(sky_map), coord=["G", "C"], title="", rot=(180, 0, 0),
                hold=True,
                unit=r"log($T_{Ant}$)",
                projection_type="mollweide",
                graticule=True,
                graticule_labels=True,
                fig=fig,
                longitude_grid_spacing=spacing,
                cmap=cmap,
                fontsize=dict(cbar_label = "large",
                              cbar_tick_label="large")

                )

    labels=np.array([ str(int(hr))+"h" for hr in 24*np.array(1-spacingfrac*np.arange(1,25/(spacingfrac*24)-1,1))])

    plt.xticks(ticks=plt.xticks()[0], labels=labels, size="large")
    plt.yticks(size="large")
    plt.title(f"Galaxy in Equatorial coordinates at {freq} MHz", size="large")
    plt.xlabel("Right Ascension", size="large")
    plt.ylabel("Declination", size="large")
    lat = np.linspace(-180, 180, 100)       
    lon = np.ones_like(lat)            
    lon *= dec_cut
#    hp.projplot(lat, lon, lonlat=True, lw=4., ls="dashed" , label="RNO-G field of view")
    plt.plot([-np.pi, np.pi], [np.radians(dec_cut), np.radians(dec_cut)], lw=4., ls="dashed" , label="RNO-G field of view")
    plt.legend(loc = "upper right")
    plt.savefig("figures/POS_ICRC/galaxy_rnog.png", dpi=300, bbox_inches="tight")
