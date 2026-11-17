import astropy.units as u
import numpy as np
from astropy.wcs import WCS


def add_wcs_compass(ax, header) -> None:
    """Label equatorial north/east on a diagnostic in native science pixels."""
    wcs = header.celestial if isinstance(header, WCS) else WCS(header).celestial
    if not wcs.has_celestial:
        return
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
    center = wcs.pixel_to_world(x, y)
    for label, angle in (("N", 0), ("E", 90)):
        end = center.directional_offset_by(angle * u.deg, 10 * u.arcsec)
        px, py = wcs.world_to_pixel(end)
        delta = np.array([(px - x) / (xmax - xmin), (py - y) / (ymax - ymin)])
        length = np.linalg.norm(delta)
        if not np.isfinite(length) or length == 0:
            continue
        point = np.array([0.85, 0.85]) + delta / length * 0.09
        ax.annotate(
            "",
            xy=point,
            xytext=(0.85, 0.85),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "cyan"},
        )
        ax.text(
            *point,
            label,
            transform=ax.transAxes,
            color="cyan",
            fontsize=9,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
        )
