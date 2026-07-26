import os
import subprocess


def solve_field(
    input_file: str,
    ra: float | None,
    dec: float | None,
    scale_low: float | None,
    scale_high: float | None,
    search_radius: float = 2,
):
    """Execute the `solve-field` command to compute a WCS solution.

    To search around a point, define both ``ra`` and ``dec``.


    Parameters
    ----------
    input_file : str
        The image file name to calibrate.

    ra: float or None
        Right ascension guess. (degrees)

    dec: float or None
        Declination guess. (degrees)

    scale_low : float
        Lower limit on pixel scale solution.  (arcsec/pix)

    scale_high : float
        Upper limit on pixel scale solution.  (arcsec/pix)

    search_radius : float, optional
        Search radius around (ra, dec).  Ignored if ``ra`` or ``dec`` is
        ``None``.  (degrees)


    Returns
    -------
    success : bool
        True if the solve-field command succeeded or file already exists.

    """

    config_file = os.environ.get("ASTROMETRY_CONFIG")
    if config_file is None:
        raise RuntimeError(
            "ASTROMETRY_CONFIG is not set. This is required to run solve-field."
        )

    command = [
        "solve-field",
        "--overwrite",
        "--config",
        config_file,
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
        "--downsample",
        "1",
    ]

    if ra is not None and dec is not None:
        command.extend(
            ["--ra", str(ra), "--dec", str(dec), "--radius", str(search_radius)]
        )

    command.append(input_file)

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"solve-field failed: {e}")
