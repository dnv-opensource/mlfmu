import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import mlfmu.utils.builder as builder

__ALL__ = ["run", "MlFmuProcess"]

logger = logging.getLogger(__name__)


class MlFmuCommand(Enum):
    BUILD = "build"
    GENERATE = "generate-code"
    COMPILE = "build-code"


def run(
    command: MlFmuCommand,
    logger: logging.Logger,
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
        logger=logger,
    )
    process.run()

    return


class MlFmuProcess:
    """Top level class encapsulating the mlfmu process."""

    command: MlFmuCommand
    fmu_name: Optional[str] = None
    build_folder: Optional[Path] = None
    source_folder: Optional[Path] = None
    ml_model_file: Optional[Path] = None
    interface_file: Optional[Path] = None
    fmu_output_folder: Optional[Path] = None
    logger: logging.Logger

    def __init__(
        self,
        command: MlFmuCommand,
        logger: logging.Logger,
        source_folder: Optional[Path] = None,
        ml_model_file: Optional[Path] = None,
        interface_file: Optional[Path] = None,
        fmu_output_folder: Optional[Path] = None,
    ):
        self._run_number: int = 0
        self._max_number_of_runs: int = 1
        self.terminate: bool = False

        self.logger = logger

        self.command = command
        self.fmu_name = None

        self.ml_model_file = ml_model_file
        self.interface_file = interface_file
        self.fmu_output_folder = fmu_output_folder

        self.source_folder = source_folder

        return

    def _build_fmu(self):
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
            # TODO: cleanup
            return

        self.fmu_name = fmi_model.name
        builder.build_fmu(
            fmu_src_path=self.source_folder / self.fmu_name,
            fmu_build_path=self.build_folder,
            fmu_save_path=self.fmu_output_folder,
        )

        # TODO: Clean up generated source and build files

        pass

    def _generate_code(self):
        # TODO: Raise errors
        if self.source_folder is None:
            raise
        if self.ml_model_file is None:
            raise
        if self.interface_file is None:
            raise
        try:
            _ = builder.generate_fmu_files(self.source_folder, self.ml_model_file, self.interface_file)
        except Exception as e:
            print(e)
            # TODO: Clean up
            return

    def _build_source_code(self):
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

        # TODO: Clean up generated source and build files
        pass

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
