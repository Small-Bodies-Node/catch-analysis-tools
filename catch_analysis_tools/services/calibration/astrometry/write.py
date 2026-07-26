from astropy.io import fits


def write_astrometry_output(
    data,
    wcs_solution,
    output_fits,
):
    """Merge the new WCS with the original FITS file data."""

    primary_hdu = fits.PrimaryHDU(
        data=data,
        header=wcs_solution.to_header(relax=True),
    )

    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(output_fits, overwrite=True)
