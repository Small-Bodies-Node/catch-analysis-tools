from .count_complete_index_files import count_complete_index_files


def index_files_complete(expected_files, verify_checksum=False):
    """Return whether all expected index files are present, plus the count."""
    files_present = count_complete_index_files(
        expected_files, verify_checksum=verify_checksum
    )
    return files_present == len(expected_files), files_present
