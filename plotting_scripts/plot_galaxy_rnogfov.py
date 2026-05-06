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

    nside = hp.get_nside(sky_map)
    npix = len(sky_map)

    # Get theta (colatitude) and phi
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Convert theta to declination
    # theta = 0 at north pole → dec = +90 deg
    dec = 0.5 * np.pi - theta  # radians

#    local_coord = get_local_coordinates((site_lat, site_lon), time, n_side=512)
#    mask = local_coord.dec.rad > dec_cut
#    mask = mask > 0

    

    plt.style.use("astroparticle_physics")
#    hp.mollview(np.log2(sky_map), coord=["G", "C"], title="Galactic noise temperature as seen from RNO-G's location", rot=(180, 0, 0),
#                hold=True,
#                unit=r"log($T_{Ant}$)",
#                )
    spacing = 60
    spacingfrac = spacing/360
    cmap = plt.get_cmap()
    fig, ax = plt.subplots()
    hp.projview(np.log10(sky_map), coord=["G", "C"], title="", rot=(180, 0, 0),
                hold=True,
                unit=r"$\log_{10} \left(T_{\text{brightness}} / K \right)$",
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
#    plt.xlabel("Right Ascension", size="large")
    plt.ylabel("Declination", size="large")
    lat = np.linspace(-180, 180, 100)       
    lon = np.ones_like(lat)            
    lon *= dec_cut
#    hp.projplot(lat, lon, lonlat=True, lw=4., ls="dashed" , label="RNO-G field of view")
#    line_fov = plt.plot([-np.pi, np.pi], [np.radians(dec_cut), np.radians(dec_cut)], lw=4., ls="dashed" , label="RNO-G field of view")
    plt.fill_between([-np.pi, np.pi], [-np.pi/2., -np.pi/2.], [np.radians(dec_cut), np.radians(dec_cut)], alpha=0.8, color="gray")
    dy = np.radians(8)
    nr_arrows = 5
#    for x in np.linspace(-0.9*np.pi, 0.9*np.pi, nr_arrows):
#        plt.arrow(x=x, y=np.radians(dec_cut)-dy/2,
#                  dx=0., dy=dy,
#                  lw=3.,
#                  head_width=np.radians(2),
#                  color = line_fov[0].get_color() 
#                  )
#    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig("figures/paper/galaxy_rnog.png", dpi=300, bbox_inches="tight")
