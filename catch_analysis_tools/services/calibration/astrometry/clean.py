import os


def cleanup_files(file_base):
    """
    Remove temporary files generated during the processing pipeline.


    Parameters
    ----------
    file_base : str
        Base filename (without extension) for the files to remove.


    Returns
    -------
    None

    """

    extensions = [
        ".axy",
        ".corr",
        ".match",
        ".new",
        ".rdls",
        ".solved",
        "-ngc.png",
        "-objs.png",
        "-indx.png",
        "-indx.xyls",
    ]
    for ext in extensions:
        fname = f"{file_base}{ext}"
        if os.path.exists(fname):
            os.remove(fname)
