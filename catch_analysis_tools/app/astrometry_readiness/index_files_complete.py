from catch_analysis_tools.app.astrometry_readiness.count_complete_index_files import (
    count_complete_index_files,
)


def index_files_complete(expected_files):
    """Return whether all expected index files are present, plus the count."""
    files_present = count_complete_index_files(expected_files)
    return files_present == len(expected_files), files_present
