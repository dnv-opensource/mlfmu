from __future__ import annotations

import logging
import os
import tempfile
from enum import Enum
from pathlib import Path

from mlfmu.utils import builder
from mlfmu.utils.path import find_default_file

__ALL__ = ["run", "MlFmuProcess"]

logger = logging.getLogger(__name__)


class MlFmuCommand(Enum):
    """Enum class for the different commands in the mlfmu process."""

    BUILD = "build"
    GENERATE = "codegen"
    COMPILE = "compile"

    @staticmethod
    def from_string(command_string: str) -> MlFmuCommand | None:
        matches = [command for command in MlFmuCommand if command.value == command_string]
        return matches[0] if matches else None


# run for mlfmu
def run(
    command: MlFmuCommand,
    interface_file: str | None,
    model_file: str | None,
    fmu_path: str | None,
    source_folder: str | None,
) -> None:
    """Run the mlfmu process.

    Run the mlfmu process with the given command and optional parameters.

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
    """Builder for creating FMUs (Functional Mock-up Units) from machine learning models.

    This class is core to executing the different commands in the mlfmu process.
    """

    def __init__(
        self,
        fmu_name: str | None = None,
        interface_file: Path | None = None,
        ml_model_file: Path | None = None,
        source_folder: Path | None = None,
        fmu_output_folder: Path | None = None,
        build_folder: Path | None = None,
        root_directory: Path | None = None,
    ) -> None:
        """Initialize a new instance of the MlFmuBuilder class.

        Parameters
        ----------
        fmu_name : str | None, optional
            The name of the FMU., by default None
        interface_file : Path | None, optional
            The path to the interface JSON file., by default None
        ml_model_file : Path | None, optional
            The path to the machine learning model file., by default None
        source_folder : Path | None, optional
            The folder containing the source code for the FMU., by default None
        fmu_output_folder : Path | None, optional
            The folder where the built FMU will be saved., by default None
        build_folder : Path | None, optional
            The folder where the FMU will be built., by default None
        root_directory : Path | None, optional
            The root directory for the builder., by default None
        """
        self.fmu_name: str | None = fmu_name
        self.interface_file: Path | None = interface_file
        self.ml_model_file: Path | None = ml_model_file
        self.source_folder: Path | None = source_folder
        self.fmu_output_folder: Path | None = fmu_output_folder
        self.build_folder: Path | None = build_folder
        self.root_directory: Path = root_directory or Path.cwd()
        #: The temporary folder used for building the FMU.
        self.temp_folder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="mlfmu_")
        self.temp_folder_path: Path = Path(self.temp_folder.name)
        logger.debug(f"Created temp folder: {self.temp_folder_path}")

    def build(self) -> None:
        """Build an FMU from the machine learning model file and interface file.

        Builds an FMU from ml_model_file and interface_file and saves it to fmu_output_folder.
        If the paths to the necessary files and directories are not given the function
        will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if ml_model_file or interface_file do not exists or is not set and cannot be easily inferred.
        ---
        """
        # sourcery skip: class-extract-method
        logger.debug("MLFmuBuilder: Start build")
        # specify folders and filenames for building
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
                "No interface json file was provided and no obvious interface file found in current working directory."
            )
        if not self.interface_file.exists():
            raise FileNotFoundError(f"The given interface json file (={self.interface_file}) does not exist.")

        self.build_folder = self.build_folder or self.default_build_folder()
        self.fmu_output_folder = self.fmu_output_folder or self.default_fmu_output_folder()

        # create fmu files
        try:
            fmi_model = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
        except Exception:
            logger.exception("Exception when running generate_fmu_files: %s")

        self.fmu_name = fmi_model.name
        builder.build_fmu(
            fmu_src_path=self.source_folder / self.fmu_name,
            fmu_build_path=self.build_folder,
            fmu_save_path=self.fmu_output_folder,
        )
        logger.debug("MLFmuBuilder: Done with build")

    def generate(self) -> None:
        """Generate C++ source code and model description from the machine learning model file and interface file.

        Generates FMU C++ source code and model description from ml_model_file and interface_file
        and saves it to source_folder.

        If the paths to the necessary files and directories are not given the function
        will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if ml_model_file or interface_file do not exists or is not set and cannot be easily inferred.
        """
        logger.debug("MLFmuBuilder: Start generate")
        # specify folders and filenames for generating
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
                "No interface json file was provided and no obvious interface file found in current working directory."
            )
        if not self.interface_file.exists():
            raise FileNotFoundError(f"The given interface json file (={self.interface_file}) does not exist.")

        # create fmu files
        try:
            fmi_model = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
            self.fmu_name = fmi_model.name
        except Exception:
            logger.exception("Exception when running generate_fmu_files: %s")

        logger.debug("MLFmuBuilder: Done with generate")

    def compile(self) -> None:
        """Compile the FMU from the C++ source code and model description.

        Compiles the FMU from the FMU C++ source code and model description contained in source_folder
        and saves it to fmu_output_folder.

        If the paths to the necessary files and directories are not given the function
        will try to find files and directories that match the ones needed.

        Raises
        ------
        FileNotFoundError
            if source_folder or fmu_name is not set and cannot be easily inferred.
        """
        logger.debug("MLFmuBuilder: Start compile")
        self.build_folder = self.build_folder or self.default_build_folder()

        self.fmu_output_folder = self.fmu_output_folder or self.default_fmu_output_folder()

        if self.fmu_name is None or self.source_folder is None:
            source_child_folder = self.default_compile_source_folder()
            if source_child_folder is None:
                raise FileNotFoundError(
                    "No valid FMU source directory found anywhere inside the current working directory "
                    f"or any given source path (={self.source_folder})."
                )
            self.fmu_name = source_child_folder.stem
            self.source_folder = source_child_folder.parent

        try:
            builder.build_fmu(
                fmu_src_path=self.source_folder / self.fmu_name,
                fmu_build_path=self.build_folder,
                fmu_save_path=self.fmu_output_folder,
            )
            logger.debug("MLFmuBuilder: Done with build (via compile)")
        except Exception:
            logger.exception("Error while running build_fmu: %s")
        logger.debug("MLFmuBuilder: Done with compile")

    def default_interface_file(self) -> Path | None:
        """Return the path to a interface json file inside self.root_directory if it can be inferred.

        Returns
        -------
        Path | None
            the path to the interface json file, or None.
        """
        return find_default_file(self.root_directory, "json", "interface")

    def default_model_file(self) -> Path | None:
        """Return the path to a ml model file inside self.root_directory if it can be inferred.

        Returns
        -------
        Path | None
            the path to the ml model file, or None.
        """
        return find_default_file(self.root_directory, "onnx", "model")

    def default_build_folder(self) -> Path:
        """Return the path to a build folder inside the temp_folder. Creates the temp_folder if it is not set.

        Returns
        -------
        Path
            the path to the build folder.
        """
        return self.temp_folder_path / "build"

    def default_build_source_folder(self) -> Path:
        """Return the path to a src folder inside the temp_folder. Creates the temp_folder if it is not set.

        Returns
        -------
        Path
            the path to the src folder.
        """
        return self.temp_folder_path / "src"

    def default_generate_source_folder(self) -> Path:
        """Return the path to the default source folder for the generate process.

        Returns
        -------
        Path
            the path to the default source folder for the generate process.
        """
        return self.root_directory

    def default_compile_source_folder(self) -> Path | None:
        """Return the path to the default source folder for the compile process.

        Searches inside self.source_folder and self.root_directory for a folder that contains a folder structure
        and files that is required to be valid ml fmu source code.

        Returns
        -------
        Path
            the path to the default source folder for the compile process.
        """
        search_folders: list[Path] = []
        if self.source_folder is not None:
            search_folders.append(self.source_folder)
        search_folders.append(self.root_directory)
        source_folder: Path | None = None
        # If source folder is not provided, try to find one in current folder that is compatible with the tool
        # I.e a folder that contains everything needed for compilation
        for current_folder in search_folders:
            for sub_folder, _, _ in os.walk(current_folder):
                try:
                    possible_source_folder = Path(sub_folder)
                    # If a fmu name is given and the candidate folder name does not match. Skip it!
                    if self.fmu_name is not None and possible_source_folder.stem != self.fmu_name:
                        continue
                    builder.validate_fmu_source_files(possible_source_folder)
                    source_folder = possible_source_folder
                    # If a match was found stop searching
                    break
                except Exception:
                    logger.exception("Exception when validating source folder: %s")
                    # Any folder that does not contain the correct folder structure and files
                    # needed for compilation will raise an exception
                    continue
            # If a match was found stop searching
            if source_folder is not None:
                break
        return source_folder

    def default_fmu_output_folder(self) -> Path:
        """Return the path to the default fmu output folder.

        Returns
        -------
        Path
            the path to the default fmu output folder.
        """
        return self.root_directory


class MlFmuProcess:
    """
    Represents the ML FMU process.

    This class encapsulates the functionality to run the ML FMU process in a self-terminated loop.
    It provides methods to control the execution of the process and manage the number of runs.
    """

    command: MlFmuCommand  #: The command to be executed by the process.
    builder: MlFmuBuilder  #: The builder object responsible for building the FMU.

    def __init__(
        self,
        command: MlFmuCommand,
        source_folder: Path | None = None,
        ml_model_file: Path | None = None,
        interface_file: Path | None = None,
        fmu_output_folder: Path | None = None,
    ) -> None:
        """Instantiate the MlFmuProcess.

        Parameters
        ----------
        command : MlFmuCommand
            The command to be executed by the process.
        source_folder : Path | None, optional
            The path to the source folder., by default None
        ml_model_file : Path | None, optional
            The path to the ML model file., by default None
        interface_file : Path | None, optional
            The path to the interface file., by default None
        fmu_output_folder : Path | None, optional
            The path to the FMU output folder., by default None
        """
        self._run_number: int = 0
        self._max_number_of_runs: int = 1
        self.terminate: bool = False

        self.command = command

        fmu_name: str | None = None
        build_folder: Path | None = None

        self.builder = MlFmuBuilder(
            fmu_name=fmu_name,
            interface_file=interface_file,
            ml_model_file=ml_model_file,
            source_folder=source_folder,
            fmu_output_folder=fmu_output_folder,
            build_folder=build_folder,
        )

    def run(self) -> None:
        """
        Run the mlfmu process.

        Runs the mlfmu process in a self-terminated loop.
        """
        # Run mlfmu process until termination is flagged
        while not self.terminate:
            try:
                self._run_process()
            except Exception:
                logger.exception("Exception in run_process for MlFmuProcess: %s")
                self.terminate = True
            self.terminate = self._run_number >= self._max_number_of_runs

        return

    def _run_process(self) -> None:
        """Execute a single run of the mlfmu process."""
        self._run_number += 1

        logger.debug(f"MlFmuProcess: Start run {self._run_number}")

        if self.command == MlFmuCommand.BUILD:
            self.builder.build()
        elif self.command == MlFmuCommand.GENERATE:
            self.builder.generate()
        elif self.command == MlFmuCommand.COMPILE:
            self.builder.compile()

        logger.debug(f"MlFmuProcess: Successfully finished run {self._run_number}")

        return
