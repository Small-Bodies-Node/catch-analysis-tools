import numpy as np
from astropy.coordinates import SkyCoord


def attach_sky_coordinates(source_list, wcs_solution):
    """Convert pixel coordinates to sky coordinates using a WCS.


    Parameters
    ----------
    source_list : pd.DataFrame
        Table with 'x' and 'y' pixel positions of detected sources.

    wcs_solution : astropy.wcs.WCS
        World coordinate system solution object.


    Returns
    -------
    source_list : pd.DataFrame
        Updated table including 'RA' and 'Dec' columns in degrees.

    sky_coords : astropy.coordinates.SkyCoord
        SkyCoord object with celestial coordinates of sources.

    """

    world = wcs_solution.pixel_to_world(
        np.asarray(source_list["x"], dtype=float),
        np.asarray(source_list["y"], dtype=float),
    )
    source_list["RA"] = world.ra.deg
    source_list["Dec"] = world.dec.deg
    sky_coords = SkyCoord(source_list["RA"], source_list["Dec"], unit="deg")

    return source_list, sky_coords
