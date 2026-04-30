from catch_analysis_tools.app.services.astrometry_data import (
    get_status,
    prepare_astrometry_data,
)


def main():
    prepare_astrometry_data()
    print(get_status())


if __name__ == "__main__":
    main()
