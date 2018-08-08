from astropy.io import fits
import kplr
import pickle

def get_flux_for_star(KID):
    """Loads time series data of tuples of (Time (JD), Aperture Flux (e-/s), Flux_error)
    given a star's Kepler ID"""
    client = kplr.API()
    star = client.star("{}".format(KID))
    curves = star.get_light_curves(fetch=True, short_cadence=False)
    data=[]
    for curve in curves:
        data1=fits.open(curve.filename)[1].data
        for point in data1:
            data.append((point[0],point[3],point[4])) #Time, flux, fluxerr
    return data

