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
    command: MlFmuCommand
        which command in the mlfmu process that should be run
    interface_file: Optional[str]
        the path to the file containing the FMU interface. Will be inferred if not provided.
    model_file: Optional[str]
        the path to the ml model file. Will be inferred if not provided.
    fmu_path: Optional[str]
        the path to where the built FMU should be saved. Will be inferred if not provided.
    source_folder: Optional[str]
        the path to where the FMU source code is located. Will be inferred if not provided.
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
    """Class for executing the different commands in the mlfmu process."""

    fmu_name: Optional[str] = None
    build_folder: Optional[Path] = None
    source_folder: Optional[Path] = None
    ml_model_file: Optional[Path] = None
    interface_file: Optional[Path] = None
    fmu_output_folder: Optional[Path] = None
    delete_build_folders: bool = False
    temp_folder: Optional[tempfile.TemporaryDirectory[str]] = None
    root_directory: Path

    def __init__(
        self,
        fmu_name: Optional[str] = None,
        interface_file: Optional[Path] = None,
        ml_model_file: Optional[Path] = None,
        source_folder: Optional[Path] = None,
        fmu_output_folder: Optional[Path] = None,
        build_folder: Optional[Path] = None,
        delete_build_folders: bool = False,
        root_directory: Optional[Path] = None,
    ):
        self.fmu_name = fmu_name
        self.interface_file = interface_file
        self.ml_model_file = ml_model_file
        self.source_folder = source_folder
        self.fmu_output_folder = fmu_output_folder
        self.build_folder = build_folder
        self.delete_build_folders = delete_build_folders
        self.root_directory = root_directory or Path(os.getcwd())

    def build(self):
        """
        Build an FMU from ml_model_file and interface_file and saves it to fmu_output_folder.

        If the paths to the necessary files and directories are not given the function will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if ml_model_file or interface_file do not exists or is not set and cannot be easily inferred.
        ---
        """
        self.source_folder = self.source_folder or self.default_build_source_folder()

        self.ml_model_file = self.ml_model_file or self.default_model_file()
        if self.ml_model_file is None:
            raise FileNotFoundError(
                "No model file was provided and no obvious model file found in current working directory (os.getcwd())"
            )
        if not self.ml_model_file.exists():
            raise FileNotFoundError(f"The given model file (={self.ml_model_file}) does not exist.")

        self.interface_file = self.interface_file or self.default_interface_file()
        if self.interface_file is None:
            raise FileNotFoundError(
                "No interface json file was provided and no obvious interface file found in current working directory (os.getcwd())"
            )
        if not self.interface_file.exists():
            raise FileNotFoundError(f"The given interface json file (={self.interface_file}) does not exist.")

        self.build_folder = self.build_folder or self.default_build_folder()

        self.fmu_output_folder = self.fmu_output_folder or self.default_fmu_output_folder()

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
            self.delete_temp_folder()

    def generate(self):
        """
        Generate FMU c++ source code and model description from ml_model_file and interface_file and saves it to source_folder.

        If the paths to the necessary files and directories are not given the function will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if ml_model_file or interface_file do not exists or is not set and cannot be easily inferred.
        """
        self.source_folder = self.source_folder or self.default_generate_source_folder()

        self.ml_model_file = self.ml_model_file or self.default_model_file()
        if self.ml_model_file is None:
            raise FileNotFoundError(
                "No model file was provided and no obvious model file found in current working directory (os.getcwd())"
            )
        if not self.ml_model_file.exists():
            raise FileNotFoundError(f"The given model file (={self.ml_model_file}) does not exist.")

        self.interface_file = self.interface_file or self.default_interface_file()
        if self.interface_file is None:
            raise FileNotFoundError(
                "No interface json file was provided and no obvious interface file found in current working directory (os.getcwd())"
            )
        if not self.interface_file.exists():
            raise FileNotFoundError(f"The given interface json file (={self.interface_file}) does not exist.")

        try:
            fmi_model = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
            self.fmu_name = fmi_model.name
        except Exception as e:
            print(e)
            if self.delete_build_folders:
                self.delete_source()

    def compile(self):
        """
        Compile FMU from FMU c++ source code and model description contained in source_folder and saves it to fmu_output_folder.

        If the paths to the necessary files and directories are not given the function will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if source_folder or fmu_name is not set and cannot be easily inferred.
        """
        self.build_folder = self.build_folder or self.default_build_folder()

        self.fmu_output_folder = self.fmu_output_folder or self.default_fmu_output_folder()

        if self.fmu_name is None or self.source_folder is None:
            source_child_folder = self.default_compile_source_folder()
            if source_child_folder is None:
                raise FileNotFoundError(
                    f"No valid FMU source directory found anywhere inside the current working directory or any given source path (={self.source_folder})."
                )
            self.fmu_name = source_child_folder.stem
            self.source_folder = source_child_folder.parent

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
            self.delete_temp_folder()

    def delete_source(self):
        """Delete the source folder if it exists."""
        if self.source_folder is not None and self.source_folder.exists():
            shutil.rmtree(self.source_folder)

    def delete_build(self):
        """Delete the build folder if it exists."""
        if self.build_folder is not None and self.build_folder.exists():
            shutil.rmtree(self.build_folder)

    def delete_temp_folder(self):
        """Delete the temp folder if it exists."""
        if self.temp_folder is not None and Path(self.temp_folder.name).exists():
            shutil.rmtree(Path(self.temp_folder.name))

    def default_interface_file(self):
        """Return the path to a interface json file inside self.root_directory if it can be inferred."""
        return MlFmuBuilder._find_default_file(self.root_directory, "json", "interface")

    def default_model_file(self):
        """Return the path to a ml model file inside self.root_directory if it can be inferred."""
        return MlFmuBuilder._find_default_file(self.root_directory, "onnx", "model")

    def default_build_folder(self):
        """Return the path to a build folder inside the temp_folder. Creates the temp_folder if it is not set."""
        self.temp_folder = self.temp_folder or tempfile.TemporaryDirectory(prefix="mlfmu_")
        return Path(self.temp_folder.name) / "build"

    def default_build_source_folder(self):
        """Return the path to a src folder inside the temp_folder. Creates the temp_folder if it is not set."""
        self.temp_folder = self.temp_folder or tempfile.TemporaryDirectory(prefix="mlfmu_")
        return Path(self.temp_folder.name) / "src"

    def default_generate_source_folder(self):
        """Return the path to the default source folder for the generate process."""
        return self.root_directory

    def default_compile_source_folder(self):
        """Return the path to the default source folder for the compile process.

        Searches inside self.source_folder and self.root_directory for a folder that contains a folder structure and files that is required to be valid ml fmu source code.
        """
        search_folders: List[Path] = []
        if self.source_folder is not None:
            search_folders.append(self.source_folder)
        search_folders.append(self.root_directory)
        source_folder: Optional[Path] = None
        # If source folder is not provide try to find one in current folder that is compatible with the tool
        # I.e a folder that contains everything needed for compilation
        for current_folder in search_folders:
            for dir, _, _ in os.walk(current_folder):
                try:
                    possible_source_folder = Path(dir)
                    # If a fmu name is given and the candidate folder name does not match. Skip it!
                    if self.fmu_name is not None and possible_source_folder.stem != self.fmu_name:
                        continue
                    builder.validate_fmu_source_files(possible_source_folder)
                    source_folder = possible_source_folder
                    # If a match was found stop searching
                    break
                except Exception:
                    # Any folder that does not contain the correct folder structure and files needed for compilation will raise and exception
                    continue
            # If a match was found stop searching
            if source_folder is not None:
                break
        return source_folder

    def default_fmu_output_folder(self):
        """Return the path to the default fmu output folder."""
        return self.root_directory

    @staticmethod
    def _find_default_file(dir: Path, file_extension: str, default_name: Optional[str] = None):
        """Return a file inside dir with the file extension that matches file_extension. If there are multiple matches it uses the closest match to default_name if given. Return None if there is no clear match."""
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


class MlFmuProcess:
    """Top level class encapsulating the mlfmu process."""

    command: MlFmuCommand
    builder: MlFmuBuilder

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

        self.builder = MlFmuBuilder(
            fmu_name,
            interface_file,
            ml_model_file,
            source_folder,
            fmu_output_folder,
            build_folder,
            delete_build_folders=True,
        )

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
        return
