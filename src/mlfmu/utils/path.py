import os
from pathlib import Path


def find_default_file(
    folder: Path,
    file_extension: str,
    default_name: str | None = None,
) -> Path | None:
    """Return a file inside folder with the file extension that matches file_extension.

    If there are multiple matches it uses the closest match to default_name if given.
    Return None if there is no clear match.

    Parameters
    ----------
    folder : Path
        the folder to search in
    file_extension : str
        file extension to search for
    default_name : str | None, optional
        file name used to determine "closest match"
        in case multiple files match file_extension, by default None

    Returns
    -------
    Path | None
        the path to the file if it is found, otherwise None
    """
    # Check if there is a file with correct file extension in current working directory. If it exists use it.
    matching_files: list[Path] = []

    for file in os.listdir(folder):
        file_path = folder / file
        if file_path.is_file() and file_path.suffix.lstrip(".") == file_extension:
            matching_files.append(file_path)

    if not matching_files:
        return None

    if len(matching_files) == 1:
        return matching_files[0]

    # If there are more matches on file extension. Use the one that matches the default name
    if default_name is None:
        return None

    name_matches = [file for file in matching_files if default_name in file.stem]

    if not name_matches:
        return None

    if len(name_matches) == 1:
        return name_matches[0]

    # If more multiple name matches use the exact match if it exists
    name_exact_matches = [file for file in matching_files if default_name == file.stem]

    return name_matches[0] if len(name_exact_matches) == 1 else None
