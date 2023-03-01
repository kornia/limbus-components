"""Script to release a new version of the components in the package."""
import argparse
from typing import List, Union
import logging
from pathlib import Path
import shutil
import fileinput
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_imports_and_save_init(dst_path: Path, src_path: Path) -> None:
    # concat files
    lines = []
    imported = []
    if dst_path.exists():
        with open(str(dst_path), 'r') as file:
            for line in file:
                # list all thw components and modules already imported
                match = re.match(r"^from .(\w*) import (\w*(?:, \w+)*)\n", line)
                assert match
                imported.append(match.group(2))
                lines.append(line)
    with open(str(dst_path), 'a') as outfile, fileinput.input(str(src_path)) as infile:
        for line in infile:
            if line == "" or line[0] == "#":
                # skip comments and empty lines
                continue
            # check/convert all imports following the patterns:
            #    from . import xyz  # for modules
            #    from .xyz import Xyz  # for Components
            match = re.match(r"^from .(\w*) import (\w*(?:, \w+)*)\n", line)
            if not match:
                raise ValueError(f"Line {line} in {dst_path} is not a valid import "
                                 f"statement.")
            else:
                imports = match.group(2).split(", ")
                for imp in imports:
                    # separate import statements
                    new_line = f"from .{match.group(1)} import {imp}\n"
                    if new_line in lines:
                        # if the exact same line is already in the file, we skip it
                        continue
                    # if the new line is not in lines means that the import cannot be in the file
                    if imp in imported:
                        raise ValueError(f"Component {imp} is already imported in "
                                         f"{dst_path}.")
                    outfile.write(new_line)


def _copy_files(src_path: Path, dest_path: Path):
    """Copy files from src_path to dest_path."""
    for elem in src_path.iterdir():
        if elem.is_file():
            if elem.name == "__init__.py":
                _parse_imports_and_save_init(dest_path / elem.name, elem)
            elif (dest_path / elem.name).exists():
                logger.warning(f"File {elem.name} already exists in {dest_path}. NOT COPIED!!!")
            else:
                shutil.copy(str(elem), str(dest_path))
        else:
            if elem.name != "__pycache__" and elem.name[0] != ".":
                new_path = dest_path / elem.name
                new_path.mkdir(exist_ok=True, parents=True)
                _copy_files(elem, new_path)


def main(paths: Union[str, List[str]]):
    """Copy components to the release path."""
    logger.info("Releasing components:")
    if isinstance(paths, str):
        paths = [paths]
    paths.append(str(Path(__file__).parent / "../limbus_components_dev"))
    dest_path = Path(__file__).parent / "../limbus_components"
    if dest_path.exists():
        logger.warning(f"Destination path {dest_path} already exists!!! Proceed with caution.")
    dest_path.mkdir(exist_ok=True, parents=True)
    for path in paths:
        logger.info(f"    {path}")
        _copy_files(Path(path), dest_path)
    logger.info("Done.")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--paths', '-p', help='List of paths to 3rd party folders with components.', required=False)
    ARGS = PARSER.parse_args()
    main(ARGS.paths if ARGS.paths else [])