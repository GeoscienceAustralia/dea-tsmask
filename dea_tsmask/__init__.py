from .impl import tsmask_temporal as tsmask_temporal_prim
from .impl import spatial_noise_filter as spatial_noise_filter_prim


DEFAULT_THRESHOLDS = dict(brightness=0.45,
                          thd=0.05,
                          dwi=-0.05,
                          cloud2=0.05,
                          land_cloud=-0.38,
                          avi=0.06,
                          wtd=-0.2,
                          cspk=0.63,
                          sspk=0.63,
                          cloud=0.14,
                          shadow=0.055)


def tsmask_temporal(avg, mndwi, msavi, thresholds=DEFAULT_THRESHOLDS):
    """
    The temporal part of the DEA timeseries cloud/shadow classification algorithm.
    No spatial noise filtering performed.
    """
    return tsmask_temporal_prim(avg, mndwi, msavi, thresholds)


def spatial_noise_filter(mask):
    """
    Apply spatial noise filter.
    """
    return spatial_noise_filter_prim(mask)


def tsmask(avg, mndwi, msavi, thresholds=DEFAULT_THRESHOLDS):
    """
    The DEA timeseries cloud/shadow classification algorithm.
    Spatial noise filter is applied.
    """
    mask = tsmask_temporal(avg, mndwi, msavi, thresholds)
    return spatial_noise_filter(spatial_noise_filter(mask))
