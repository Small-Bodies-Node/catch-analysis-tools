import glob
import os

from .get_index_dir import get_index_dir
from .index_manifest import file_matches_entry


def count_complete_index_files(expected_files=None, verify_checksum=False):
    """Count local files matching the expected manifest entries."""
    if expected_files is None:
        files = glob.glob(str(get_index_dir() / "index-*.fits"))
        return len(
            [
                path
                for path in files
                if os.path.exists(path) and os.path.getsize(path) > 0
            ]
        )

    return sum(
        file_matches_entry(get_index_dir(), entry, verify_checksum)
        for entry in expected_files
    )
