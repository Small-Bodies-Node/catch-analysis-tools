import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def plot_color_correction(
    color_mags: np.ndarray,
    calibrated_magnitude: np.ndarray,
    instrumental_magnitude: np.ndarray,
    zero_point: float,
    color_term: float,
    color_index: str,
):
    """
    Plot the relation between instrumental and calibrated magnitudes.


    Parameters
    ----------
    color_mags : array_like
        Color indices (obs_band - cal_band) of matched stars.

    calibrated_magnitude : array_like
        Calibrated magnitudes from reference catalog.

    instrumental_magnitude : array_like
        Instrumental magnitudes measured.

    zero_point : float
        Photometric zero-point magnitude.

    color_term : float
        Color term coefficient.

    color_index : str
        Label for the color axis (e.g. "g-r").


    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects for the plot.

    """

    fig, ax = plt.subplots()

    ax.scatter(
        color_mags,
        calibrated_magnitude - instrumental_magnitude,
        marker=".",
    )

    x = np.linspace(0, 1.5, 100)
    ax.plot(
        x,
        color_term * x + zero_point,
        color="red",
        label=f"$m = C\\times({color_index}) + ZP$",
    )

    ax.set_xlabel(f"${color_index}$ (mag)")
    ax.set_ylabel(r"$m - m_{\mathrm{inst}}$ (mag)")
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_photometric_matches(
    image_sub: np.ndarray,
    source_list: pd.DataFrame,
    matched_idx,
    color_corrected_idx,
):
    """
    Overlay detected and matched sources on the background-subtracted image.


    Parameters
    ----------

    image_sub : np.ndarray
        Background-subtracted image.

    source_list : pd.DataFrame
        Table of detected sources with "x" and "y" pixel positions.

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

    mean_value = np.mean(image_sub)
    std_value = np.std(image_sub)

    im = ax.imshow(
        image_sub,
        interpolation="nearest",
        origin="lower",
        cmap="gray",
    )
    im.set_clim(vmin=mean_value - std_value, vmax=mean_value + std_value)

    fig.colorbar(im, ax=ax)

    ax.plot(
        source_list["x"],
        source_list["y"],
        "+",
        markersize=5,
        label="Detected",
        color="red",
    )

    ax.plot(
        source_list["x"].iloc[matched_idx],
        source_list["y"].iloc[matched_idx],
        "o",
        markersize=10,
        color="blue",
        markerfacecolor="none",
        label="Matched",
    )

    ax.plot(
        source_list["x"].iloc[color_corrected_idx],
        source_list["y"].iloc[color_corrected_idx],
        "o",
        markersize=15,
        color="yellow",
        markerfacecolor="none",
        label="Selected for Color Correction",
    )

    ax.legend()

    return fig, ax
