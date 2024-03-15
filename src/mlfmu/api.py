import logging
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Optional

import mlfmu.utils.builder as builder

__ALL__ = ["run", "MlFmuProcess"]

logger = logging.getLogger(__name__)


class MlFmuCommand(Enum):
    BUILD = "build"
    GENERATE = "codegen"
    COMPILE = "compile"

    @staticmethod
    def from_string(command_string: str):
        matches = [command for command in MlFmuCommand if command.value == command_string]
        if len(matches) == 0:
            return None
        return matches[0]


def run(
    command: MlFmuCommand,
    interface_file: Optional[str],
    model_file: Optional[str],
    fmu_path: Optional[str],
    source_folder: Optional[str],
):
    """Run the mlfmu process.

    Run the mlfmu process and .. (long description).

    Parameters
    ----------

    Raises
    ------

    """

    process = MlFmuProcess(
        command=command,
        source_folder=Path(source_folder) if source_folder is not None else None,
        interface_file=Path(interface_file) if interface_file is not None else None,
        ml_model_file=Path(model_file) if model_file is not None else None,
        fmu_output_folder=Path(fmu_path) if fmu_path is not None else None,
    )
    process.run()

    return


class MlFmuBuilder:
    fmu_name: Optional[str] = None
    build_folder: Optional[Path] = None
    source_folder: Optional[Path] = None
    ml_model_file: Optional[Path] = None
    interface_file: Optional[Path] = None
    fmu_output_folder: Optional[Path] = None
    delete_build_folders: bool = False

    def __init__(
        self,
        fmu_name: Optional[str] = None,
        interface_file: Optional[Path] = None,
        ml_model_file: Optional[Path] = None,
        source_folder: Optional[Path] = None,
        fmu_output_folder: Optional[Path] = None,
        build_folder: Optional[Path] = None,
        delete_build_folders: bool = False,
    ):
        self.fmu_name = fmu_name
        self.interface_file = interface_file
        self.ml_model_file = ml_model_file
        self.source_folder = source_folder
        self.fmu_output_folder = fmu_output_folder
        self.build_folder = build_folder
        self.delete_build_folders = delete_build_folders

    def build(self):
        # TODO: Raise errors
        if self.source_folder is None:
            raise
        if self.ml_model_file is None:
            raise
        if self.interface_file is None:
            raise

        if self.build_folder is None:
            raise
        if self.fmu_output_folder is None:
            raise

        try:
            fmi_model = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
        except Exception as e:
            print(e)
            if self.delete_build_folders:
                self.delete_source()
            return

        self.fmu_name = fmi_model.name
        builder.build_fmu(
            fmu_src_path=self.source_folder / self.fmu_name,
            fmu_build_path=self.build_folder,
            fmu_save_path=self.fmu_output_folder,
        )

        if self.delete_build_folders:
            self.delete_source()
            self.delete_build()
        pass

    def generate(self):
        # TODO: Raise errors
        if self.source_folder is None:
            raise
        if self.ml_model_file is None:
            raise
        if self.interface_file is None:
            raise
        try:
            fmi_model = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
            self.fmu_name = fmi_model.name
        except Exception as e:
            print(e)
            if self.delete_build_folders:
                self.delete_source()
            return

    def compile(self):
        if self.source_folder is None:
            raise
        if self.build_folder is None:
            raise
        if self.fmu_output_folder is None:
            raise
        if self.fmu_name is None:
            raise

        try:
            builder.build_fmu(
                fmu_src_path=self.source_folder / self.fmu_name,
                fmu_build_path=self.build_folder,
                fmu_save_path=self.fmu_output_folder,
            )
        except Exception as e:
            print(e)

        if self.delete_build_folders:
            self.delete_build()
        pass

    def delete_source(self):
        if self.source_folder is not None and self.source_folder.exists():
            shutil.rmtree(self.source_folder)

    def delete_build(self):
        if self.build_folder is not None and self.build_folder.exists():
            shutil.rmtree(self.build_folder)


class MlFmuProcess:
    """Top level class encapsulating the mlfmu process."""

    command: MlFmuCommand
    builder: MlFmuBuilder
    temp_folder: Optional[tempfile.TemporaryDirectory[str]] = None

    def __init__(
        self,
        command: MlFmuCommand,
        source_folder: Optional[Path] = None,
        ml_model_file: Optional[Path] = None,
        interface_file: Optional[Path] = None,
        fmu_output_folder: Optional[Path] = None,
    ):
        self._run_number: int = 0
        self._max_number_of_runs: int = 1
        self.terminate: bool = False

        self.command = command

        fmu_name: Optional[str] = None
        build_folder: Optional[Path] = None

        current_folder = Path(os.getcwd())
        # For
        if self.command == MlFmuCommand.BUILD or self.command == MlFmuCommand.GENERATE:
            if interface_file is None:
                # Find a default file if it exists
                interface_file_match = self._find_default_file(current_folder, "json", "interface")
                if interface_file_match is None:
                    raise FileNotFoundError(
                        "No interface file provided and no good match found ion current working directory."
                    )
                interface_file = interface_file_match
            else:
                interface_file = interface_file

            if ml_model_file is None:
                # Check if there is a onnx file in current working directory. If it exists use it.
                model_file_match = self._find_default_file(current_folder, "onnx", "model")
                if model_file_match is None:
                    raise FileNotFoundError(
                        "No model file provided and no good match found ion current working directory."
                    )
                ml_model_file = model_file_match
            else:
                ml_model_file = ml_model_file

        if self.command == MlFmuCommand.BUILD or self.command == MlFmuCommand.COMPILE:
            self.temp_folder = self.temp_folder or tempfile.TemporaryDirectory(prefix="mlfmu_", delete=False)
            build_folder = Path(self.temp_folder.name) / "build"
            fmu_output_folder = current_folder if fmu_output_folder is None else fmu_output_folder

        if self.command == MlFmuCommand.BUILD:
            self.temp_folder = self.temp_folder or tempfile.TemporaryDirectory(prefix="mlfmu_", delete=False)
            source_folder = Path(self.temp_folder.name) / "src"
        elif self.command == MlFmuCommand.GENERATE:
            source_folder = source_folder or current_folder
        elif self.command == MlFmuCommand.COMPILE:
            if source_folder is None:
                # If source folde ris not provide try to find one in current folder that is compatible with the tool
                # I.e a folder that contains everything needed for compilation
                for dir, _, _ in os.walk(current_folder):
                    try:
                        possible_source_folder = Path(dir)
                        builder.validate_fmu_source_files(possible_source_folder)
                        source_folder = possible_source_folder
                        break
                    except Exception:
                        continue

            if source_folder is None:
                raise
            fmu_name = source_folder.stem
            source_folder = source_folder.parent

        self.builder = MlFmuBuilder(
            fmu_name,
            interface_file,
            ml_model_file,
            source_folder,
            fmu_output_folder,
            build_folder,
            delete_build_folders=False,
        )

        return

    def run(self):
        """Run the mlfmu process.

        Runs the mlfmu process in a self-terminated loop.
        """

        # Run mlfmu process until termination is flagged
        while not self.terminate:
            self._run_process()
            self.terminate = self._run_number >= self._max_number_of_runs
        return

    @property
    def run_number(self) -> int:
        """Example for a read only property."""
        return self._run_number

    @property
    def max_number_of_runs(self) -> int:
        """Example for a read/write property implemented through a pair of explicit
        getter and setter methods (see below for the related setter method).
        """
        return self._max_number_of_runs

    @max_number_of_runs.setter
    def max_number_of_runs(self, value: int):
        """Setter method that belongs to above getter method.

        Note that implementing specific getter- and setter methods is in most cases not necessary.
        The same can be achieved by simply making the instance variable a public attribute.
        I.e., declaring the instance variable in __init__() not as
        self._max_number_of_runs: int = ..  # (-> private instance variable)
        but as
        self.max_number_of_runs: int = ..   # (-> public attribute)

        However, in some cases the additional effort of implementing explicit getter- and setter- methods
        as in this example can be reasoned, for instance if you have a need for increased control
        and want be able to cancel or alter code execution, or write log messages whenever a property
        gets reads or written from outside.
        """

        self._max_number_of_runs = value
        return

    @staticmethod
    def _find_default_file(dir: Path, file_extension: str, default_name: Optional[str] = None):
        # Check if there is a file with correct file extension in current working directory. If it exists use it.
        matching_files: List[Path] = []

        for file in os.listdir(dir):
            file_path = dir / file
            if file_path.is_file() and file_path.suffix.lstrip(".") == file_extension:
                matching_files.append(file_path)

        if len(matching_files) == 0:
            return

        if len(matching_files) == 1:
            return matching_files[0]

        # If there are more matches on file extension. Use the one that matches the default name
        if default_name is None:
            return

        name_matches = [file for file in matching_files if default_name in file.stem]

        if len(name_matches) == 0:
            return

        if len(name_matches) == 1:
            return name_matches[0]

        # If more multiple name matches use the exact match if it exists
        name_exact_matches = [file for file in matching_files if default_name == file.stem]

        if len(name_exact_matches) == 1:
            return name_matches[0]
        return

    def _run_process(self):
        """Execute a single run of the mlfmu process."""
        self._run_number += 1

        logger.info(f"Start run {self._run_number}")

        logger.info(f"Successfully finished run {self._run_number}")

        if self.command == MlFmuCommand.BUILD:
            self.builder.build()
        elif self.command == MlFmuCommand.GENERATE:
            self.builder.generate()
        elif self.command == MlFmuCommand.COMPILE:
            self.builder.compile()

        if self.temp_folder is not None:
            temp_folder_path = Path(self.temp_folder.name)
            if temp_folder_path.exists():
                try:
                    shutil.rmtree(temp_folder_path)
                except Exception as e:
                    print(e)
        return


def _do_cool_stuff(run_number: int) -> str:
    """Do cool stuff.

    Converts the passed in run number to a string.

    Parameters
    ----------
    run_number : int
        the run number

    Returns
    -------
    str
        the run number converted to string
    """
    result: str = ""
    return result
