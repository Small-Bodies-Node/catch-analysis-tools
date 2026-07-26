import argparse

from ...services.calibration.photometry import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Photometric calibration of an image.")
    parser.add_argument("input_fits", help="Path to FITS image")
    parser.add_argument(
        "--snr", type=float, default=7.0, help="Detection S/N threshold (default: 7.0)"
    )
    parser.add_argument(
        "--aperture_radius",
        default=7,
        help="Photometric aperture radius",
    )
    parser.add_argument(
        "--catalog",
        default="PanSTARRS1",
        help="Photometric reference catalog (default: PanSTARRS1)",
    )
    parser.add_argument(
        "--cal_band", default="r", help="Reference catalog bandpass (default: r)"
    )
    parser.add_argument(
        "--color_index", help="Color index for color correction, e.g., g-r"
    )
    args = parser.parse_args()

    result = run_pipeline(
        args.input_fits,
        args.snr,
        args.aperture_radius,
        args.catalog,
        args.cal_band,
        args.color_index,
        return_plot=False,
    )

    print(result)


if __name__ == "__main__":
    main()
