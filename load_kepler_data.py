from astropy.io import fits
import kplr
import pickle

def get_flux_for_star(KID):
    """Loads time series data of tuples of (Time (JD), Aperture Flux (e-/s), Flux_error)
    given a star's Kepler ID
    returns a list of light curves (doesn't append them all together because of different
    baselines, each curve has its own baseline)"""
    client = kplr.API()
    star = client.star("{}".format(KID))
    curves = star.get_light_curves(fetch=True, short_cadence=False)
    data=[]
    for curve in curves:
        _data=[]
        data1=fits.open(curve.filename)[1].data
        for point in data1:
            if point[7]>0:
                _data.append((point[0],point[7],point[8])) #Time, flux, fluxerr
        data.append(_data)
    return data
