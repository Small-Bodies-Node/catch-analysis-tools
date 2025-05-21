import os
import subprocess
import json
import requests
import numpy as np
import pandas as pd
import sep
import fitsio
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import calviacat as cvc

def load_fits_image(target_name, data_sources, fn=3):
    """
    Load a FITS image from local disk or download via the CATCH API.

    Parameters
    ----------
    target_name : str
        Identifier of the astronomical target (e.g., comet or asteroid).
    data_sources : str
        Name of the survey (e.g., neat_palomar_tricam, ps1dr2, etc.).
    fn : int, optional
        Frame index to select from the API response (default is 0).

    Returns
    -------
    input_fits : str
        Path to the FITS file on local disk.
    file_base : str
        Base filename without extension.
    hdulist : astropy.io.fits.HDUList
        Opened FITS HDUList object.
    image : array_like
        2D image data array from the FITS file.

    Raises
    ------
    requests.HTTPError
        If any HTTP request fails.
    IndexError
        If the frame index `fn` is out of range in the API response.
    """
    input_fits = f"{target_name}_{data_sources}.fits"
    file_base = os.path.splitext(input_fits)[0]

    if os.path.exists(input_fits):
        hdulist = fits.open(input_fits)
        image = fitsio.read(input_fits)
    else:
        params = {"target": target_name, "sources": data_sources, "cached": "true"}
        base_url = "https://catch-dev-api.astro.umd.edu"
        res = requests.get(f"{base_url}/catch", params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if data.get('message') == 'Found cached data.  Retrieve from results URL.':
            results_url = data.get('results')
            res = requests.get(results_url, timeout=10)
            res.raise_for_status()
            data = res.json()

        try:
            entry = data['data'][fn]
        except (IndexError, KeyError):
            raise IndexError(f"Frame index {fn} out of range")

        cutout_url = entry.get('cutout_url')
        hdulist = fits.open(cutout_url)
        image = fitsio.read(cutout_url)
        hdulist.writeto(input_fits, overwrite=True)

    return input_fits, file_base, hdulist, image


def run_solve_field(input_fits, output_wcs, pixel_scale, scale_units="arcsecperpix"):
    """
    Execute the `solve-field` command to compute a WCS solution.

    Parameters
    ----------
    input_fits : str
        Path to the input FITS image.
    output_wcs : str
        Path for the output WCS solution file.
    pixel_scale : float
        Approximate pixel scale (e.g., arcsec/pixel).
    scale_units : str, optional
        Units for pixel scale (default is "arcsecperpix").

    Returns
    -------
    success : bool
        True if the solve-field command succeeded or file already exists.
    """
    if os.path.exists(output_wcs):
        print(f"Output file '{output_wcs}' already exists. Skipping solve-field execution.")          
        return True
    
    command = [
        'solve-field',
        '--overwrite',
        '--scale-units', scale_units,
        '--scale-low', str(pixel_scale * 0.5),
        '--scale-high', str(pixel_scale * 2.0),
        input_fits
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"solve-field failed: {e}")


def find_sources(image, snr, aperture_radius=7.0):
    """
    Detect sources in an image using SEP background subtraction and extraction.

    Parameters
    ----------
    image : np.ndarray or MaskedArray
        2D image data, possibly masked, for source detection.
    snr : float
        Minimum signal-to-noise ratio threshold for source extraction.
    aperture_radius : float, optional
        Radius of the circular aperture in pixels for flux summation (default is 7.0).

    Returns
    -------
    source_list : pd.DataFrame
        Table of detected sources with aperture photometry columns.
    telescope_image_sub : np.ndarray
        Background-subtracted image array.
    """
    mask_lower_limit = 0 
    mask = np.isinf(image) | (image <= mask_lower_limit)
    image = np.ma.masked_array(image, mask)
    image_data = np.asarray(
        getattr(image, "filled", lambda x: x)(0),
        dtype=np.float32
    )
    bkg = sep.Background(image_data)
    telescope_image_sub = image_data - bkg.back()
    sep.set_sub_object_limit(500)
    sources = sep.extract(
        telescope_image_sub,
        thresh=snr,
        err=bkg.globalrms,
        deblend_nthresh=16
    )
    source_list = pd.DataFrame(sources)
    flux, flux_err, _ = sep.sum_circle(
        telescope_image_sub,
        source_list['x'], source_list['y'],
        aperture_radius,
        err=bkg.globalrms
    )
    source_list['aperture_sum'] = flux
    source_list['aperture_err'] = flux_err
    source_list = source_list[source_list['aperture_sum'] > 0].reset_index(drop=True)
    return source_list, telescope_image_sub


def load_wcs(output_wcs):
    """
    Load a WCS solution from a FITS file header.

    Parameters
    ----------
    output_wcs : str
        Path to the FITS file containing the WCS header from astrometry.net().

    Returns
    -------
    wcs_solution : astropy.wcs.WCS
        World coordinate system solution object.
    """
    if not os.path.exists(output_wcs):
        raise FileNotFoundError(f"WCS file not found: {output_wcs}")
    with fits.open(output_wcs) as hdul:
        wcs_solution = WCS(hdul[0].header)
    return wcs_solution


def retrieve_sources(source_list, wcs_solution):
    """
    Convert pixel coordinates to sky coordinates using a WCS.

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
    world = wcs_solution.pixel_to_world(source_list['x'], source_list['y'])
    source_list['RA'] = [c.ra.deg for c in world]
    source_list['Dec'] = [c.dec.deg for c in world]
    sky_coords = SkyCoord(source_list['RA'], source_list['Dec'], unit='deg')
    return source_list, sky_coords


def calibrate_photometry(sky_coords, source_list, catalog='PanSTARRS1'):
    """
    Calibrate instrumental magnitudes against a Pan-STARRS1 catalog.

    Parameters
    ----------
    sky_coords : astropy.coordinates.SkyCoord
        Celestial coordinates of detected sources.
    source_list : pd.DataFrame
        Table of detected sources containing 'aperture_sum'.
    catalog : str, optional  
        Name of the photometric catalog to use (default is 'PanSTARRS1').  

    Returns
    -------
    calibration : dict
        Dictionary with keys:
        - 'zp': zero-point magnitude
        - 'C': color term coefficient
        - 'unc': uncertainty of zero-point
        - 'g': calibrated magnitudes array
        - 'g_inst': instrumental magnitudes array
        - 'gmr': color indices (g-r)
        - 'objids': matched catalog object IDs
        - 'distances': matching distances
    """
    try:
        CatalogClass = getattr(cvc, catalog)
    except AttributeError:
        raise ValueError(f"Catalog '{catalog}' not found in calviacat")
    
    ref = CatalogClass('cat.db')
    results = ref.search(sky_coords)
    if len(results[0]) < 500:
        ref.fetch_field(sky_coords)
    objids, distances = ref.xmatch(sky_coords)
    g_inst = -2.5 * np.log10(source_list['aperture_sum'].values)
    zp, C, unc, g, gmr, gmi = ref.cal_color(objids, g_inst, 'g', 'g-r')
    return {'zp': zp, 'C': C, 'unc': unc, 'g': g, 'g_inst': g_inst, 'gmr': gmr, 'objids': objids, 'distances': distances}


def plot_color_correction(gmr, g, g_inst, C, zp):
    """
    Plot the relation between instrumental and calibrated magnitudes.

    Parameters
    ----------
    gmr : array_like
        Color indices (g - r) of matched stars.
    g : array_like
        Calibrated magnitudes from reference catalog.
    g_inst : array_like
        Instrumental magnitudes measured.
    C : float
        Color term coefficient.
    zp : float
        Zero-point magnitude.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects for the plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(gmr, g - g_inst, marker='.')
    x = np.linspace(0, 1.5, 100)
    ax.plot(x, C * x + zp)
    ax.set_xlabel('$g-r$ (mag)')
    ax.set_ylabel('$g - g_{inst}$ (mag)')
    plt.tight_layout()
    return fig, ax


def plot_image(telescope_image_sub, source_list, matched_idx, colored_idx):
    """
    Overlay detected and matched sources on the background-subtracted image.

    Parameters
    ----------
    telescope_image_sub : np.ndarray
        Background-subtracted image array.
    source_list : pd.DataFrame
        Table of detected sources with 'x' and 'y' pixel positions.
    matched_idx : array_like
        Indices of matched catalog sources in source_list.
    colored_idx : array_like
        Indices of sources selected for color correction.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects for the plot.
    """
    fig, ax = plt.subplots()
    m, s = np.mean(telescope_image_sub), np.std(telescope_image_sub)
    im = ax.imshow(telescope_image_sub, interpolation='nearest', origin='lower', cmap='gray')
    im.set_clim(vmin=m-s, vmax=m+s)
    fig.colorbar(im, ax=ax)
    ax.plot(source_list['x'], source_list['y'], '+', markersize=5, label='Detected', color='red',)
    ax.plot(source_list['x'].iloc[matched_idx], source_list['y'].iloc[matched_idx], 'o', markersize=10, color='blue', markerfacecolor='none', label='Matched')
    ax.plot(source_list['x'].iloc[colored_idx], source_list['y'].iloc[colored_idx], 'o', markersize=15, color='yellow', markerfacecolor='none', label='Selected for Color Corr')
    ax.legend()

    return fig, ax


def create_header(image, wcs_solution, zp, unc, source_list, matched_idx, colored_idx, input_fits):
    """
    Create and write a FITS file with calibrated header and source tables.

    Parameters
    ----------
    image : array_like
        Original 2D image data array.
    wcs_solution : astropy.wcs.WCS
        WCS solution for the image.
    zp : float
        Zero-point magnitude.
    unc : float
        Uncertainty of the zero-point.
    source_list : pd.DataFrame
        Table of detected sources.
    matched_idx : array_like
        Indices for matched catalog sources.
    colored_idx : array_like
        Indices for color-correction sources.
    input_fits : str, optional
        Filename for the input FITS file.

    Returns
    -------
    None
    """
    image_arr = np.asarray(image)
    primary_hdu = fits.PrimaryHDU(data=image_arr, header=wcs_solution.to_header())
    primary_hdu.header['ZP'] = zp
    primary_hdu.header['ZP_STD'] = unc
    primary_hdu.header['SUV_FLT'] = 'r'
    primary_hdu.header['REF_CATA'] = 'PANSTARR'
    primary_hdu.header['REF_FLT'] = 'g'
    primary_hdu.header['COR_COR'] = 'Y'
    primary_hdu.header['CAT_COR'] = 'g-r'
    source_list_clean = source_list.applymap(lambda x: x.filled(np.nan) if hasattr(x, 'filled') else x)
    detected_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean), name='DETECTED_SOURCES')
    if not source_list_clean.empty:
        matched_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean.iloc[matched_idx].reset_index(drop=True)), name='SELECTED_STARS')
        colored_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean.iloc[colored_idx].reset_index(drop=True)), name='START_COLOR_CORRECTION')
    else:
        matched_hdu = fits.BinTableHDU(name='SELECTED_STARS')
        colored_hdu = fits.BinTableHDU(name='START_COLOR_CORRECTION')
    hdul = fits.HDUList([primary_hdu, detected_hdu, matched_hdu, colored_hdu])
    hdul.writeto(input_fits, overwrite=True)


def cleanup_files(file_base):
    """
    Remove temporary files generated during the processing pipeline.

    Parameters
    ----------
    file_base : str
        Base filename (without extension) for the files to remove.

    Returns
    -------
    None
    """
    extensions = ['.axy', '.corr', '.match', '.new', '.rdls', '.solved',
                  '-ngc.png', '-objs.png', '-indx.png', '-indx.xyls']
    for ext in extensions:
        fname = f"{file_base}{ext}"
        if os.path.exists(fname):
            os.remove(fname)


if __name__ == "__main__":
    # Configuration
    target_name  = "103P"
    data_sources = "atlas_mauna_loa"
    pixel_scale  = 1.86     # arcsec/pixel
    snr          = 7.0      # detection threshold

    input_fits, file_base, hdulist, image = load_fits_image(target_name, data_sources)

    output_wcs = f"{file_base}.wcs"
    try:
        if run_solve_field(input_fits, output_wcs, pixel_scale):
            wcs_solution = load_wcs(output_wcs)
        else:
            raise RuntimeError("solve-field did not produce a WCS solution")
    except Exception as e:
        raise SystemExit(f"WCS calibration failed: {e}")

    source_list, telescope_image_sub = find_sources(image, snr)
    source_list, sky_coords = retrieve_sources(source_list, wcs_solution)
    calibration = calibrate_photometry(sky_coords, source_list)
    zp     = calibration["zp"]
    C      = calibration["C"]
    unc    = calibration["unc"]
    g      = calibration["g"]
    g_inst = calibration["g_inst"]
    gmr    = calibration["gmr"]
    objids = calibration["objids"]

    if hasattr(objids, "mask"):
        matched_idx = np.where(~objids.mask)[0]
    else:
        matched_idx = np.arange(len(source_list))
    if hasattr(gmr, "mask"):
        colored_idx = np.where(~gmr.mask)[0]
    else:
        colored_idx = np.arange(len(source_list))

    fig1, ax1 = plot_color_correction(gmr, g, g_inst, C, zp)
    fig2, ax2 = plot_image(telescope_image_sub, source_list, matched_idx, colored_idx)
    plt.show()

    create_header(
        image, wcs_solution, zp, unc,
        source_list, matched_idx, colored_idx, input_fits
    )

    cleanup_files(file_base)