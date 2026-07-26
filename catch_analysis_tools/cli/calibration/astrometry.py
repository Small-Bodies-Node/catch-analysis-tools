import argparse

from ...services.calibration.astrometry import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Astrometric calibration for a FITS image."
    )

    parser.add_argument(
        "input_fits",
        help="Path to input FITS image.",
    )

    parser.add_argument(
        "--Ra",
        type=float,
        required=True,
        help="Initial RA estimate in degrees.",
    )

    parser.add_argument(
        "--Dec",
        type=float,
        required=True,
        help="Initial Dec estimate in degrees.",
    )

    parser.add_argument(
        "--pixel_scale",
        type=float,
        default=1.86,
        help="Pixel scale in arcsec/pixel. Default: 1.86.",
    )

    parser.add_argument(
        "--search_radius",
        type=float,
        default=2,
        help="Search radius around Ra, Dec in degrees. Default: 2.",
    )

    parser.add_argument(
        "--output_fits",
        default=None,
        help="Output astrometrically calibrated FITS file.",
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.input_fits,
            args.Ra,
            args.Dec,
            True,
            args.pixel_scale,
            search_radius=args.search_radius,
            output_file=args.output_fits,
        )

        print(f"Astrometric calibration complete: {result['output_fits']}")

    except Exception as exc:
        raise SystemExit(f"Astrometric calibration failed: {exc}")


if __name__ == "__main__":
    main()
