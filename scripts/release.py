"""Script to release a new version of the components in the package."""
import argparse
from typing import List, Union
import logging
from pathlib import Path
import shutil
import fileinput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _copy_files(src_path: Path, dest_path: Path):
    """Copy files from src_path to dest_path."""
    for elem in src_path.iterdir():
        if elem.is_file():
            if (dest_path / elem.name).exists():
                if elem.name == "__init__.py":
                    # concat files
                    with open(str(dest_path / elem.name), 'a') as outfile, fileinput.input(str(elem)) as infile:
                        for line in infile:
                            outfile.write(line)
                else:
                    logger.warning(f"File {elem.name} already exists in {dest_path}. NOT COPIED!!!")
            else:
                shutil.copy(elem, dest_path)
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