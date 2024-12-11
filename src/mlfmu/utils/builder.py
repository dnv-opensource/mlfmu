import logging
import os
import shutil
import subprocess
from pathlib import Path

from pydantic import ValidationError

from mlfmu.types.fmu_component import FmiModel, ModelComponent
from mlfmu.types.onnx_model import ONNXModel
from mlfmu.utils.fmi_builder import generate_model_description
from mlfmu.utils.signals import range_list_expanded

# Paths to files needed for build
path_to_this_file = Path(__file__).resolve()
absolute_path = path_to_this_file.parent.parent
fmu_build_folder = absolute_path / "fmu_build"
template_parent_path = fmu_build_folder / "templates" / "fmu"

logger = logging.getLogger(__name__)


def format_template_file(
    template_path: Path,
    save_path: Path,
    data: dict[str, str],
) -> None:
    """
    Replace all the template strings with their corresponding values and save to a new file.

    Args
    ----
        template_path (Path): The path to the template file.
        save_path (Path): The path to save the formatted file.
        data (dict[str, str]): The data containing the values to replace in the template.
    """
    # TODO: Need to check that these calls are safe from a cybersecurity point of view  # noqa: TD002
    with Path.open(template_path, encoding="utf-8") as template_file:
        template_string = template_file.read()

    formatted_string = template_string.format(**data)
    with Path.open(save_path, "w", encoding="utf-8") as save_file:
        _ = save_file.write(formatted_string)


def create_model_description(
    fmu: FmiModel,
    src_path: Path,
) -> None:
    """
    Generate modelDescription.xml structure for FMU, and save it in a file.

    Args
    ----
        fmu (FmiModel): The FMI model.
        src_path (Path): The path to save the model description file.
    """
    xml_structure = generate_model_description(fmu_model=fmu)

    # Save in file
    xml_structure.write(src_path / "modelDescription.xml", encoding="utf-8")


def make_fmu_dirs(src_path: Path) -> None:
    """
    Create all the directories needed to put all the FMU files in.

    Args
    ----
        src_path (Path): The path to the FMU source directory.
    """
    sources_path = src_path / "sources"
    resources_path = src_path / "resources"
    sources_path.mkdir(parents=True, exist_ok=True)
    resources_path.mkdir(parents=True, exist_ok=True)


def create_files_from_templates(
    data: dict[str, str],
    fmu_src: Path,
) -> None:
    """
    Create and format all needed C++ files for FMU generation.

    Args
    ----
        data (dict[str, str]): The data containing the values to format the template files.
        fmu_src (Path): The path to the FMU source directory.
    """
    sources_path = fmu_src / "sources"
    file_names = ["fmu.cpp", "model_definitions.h"]

    paths = [
        (
            template_parent_path / "_template.".join(file_name.split(".")),
            sources_path / file_name,
        )
        for file_name in file_names
    ]

    for template_path, save_path in paths:
        format_template_file(template_path, save_path, data)


def format_template_data(onnx: ONNXModel, fmi_model: FmiModel, model_component: ModelComponent) -> dict[str, str]:
    """
    Generate the key-value pairs needed to format the template files to valid C++.

    Args
    ----
        onnx (ONNXModel): The ONNX model.
        fmi_model (FmiModel): The FMI model.
        model_component (ModelComponent): The model component.

    Returns
    -------
        dict[str, str]: The formatted template data.
    """
    # Work out template mapping between ONNX and FMU ports
    inputs, outputs, state_init = fmi_model.get_template_mapping()
    state_output_indexes = [
        index for state in model_component.states for index in range_list_expanded(state.agent_output_indexes)
    ]

    # Total number of inputs/outputs/internal states
    num_fmu_variables = fmi_model.get_total_variable_number()
    num_fmu_inputs = len(inputs)
    num_fmu_outputs = len(outputs)
    num_onnx_states = len(state_output_indexes)
    num_onnx_state_init = len(state_init)

    # Checking compatibility between ModelComponent and ONNXModel
    if num_fmu_inputs > onnx.input_size:
        raise ValueError(
            "The number of total input indexes for all inputs and parameter in the interface file "
            f"(={num_fmu_inputs}) cannot exceed the input size of the ml model (={onnx.input_size})"
        )

    if num_fmu_outputs > onnx.output_size:
        raise ValueError(
            "The number of total output indexes for all outputs in the interface file "
            f"(={num_fmu_outputs}) cannot exceed the output size of the ml model (={onnx.output_size})"
        )
    if num_onnx_states > onnx.state_size:
        raise ValueError(
            "The number of total output indexes for all states in the interface file "
            f"(={num_onnx_states}) cannot exceed either the state input size (={onnx.state_size})"
        )
    if num_onnx_state_init > onnx.state_size:
        raise ValueError(
            "The number of states that are initialized in the interface file "
            f"(={num_onnx_state_init}) cannot exceed either the state input size (={onnx.state_size})"
        )

    # Flatten vectors to comply with template requirements
    # -> onnx-index, variable-reference, onnx-index, variable-reference ...
    flattened_input_string = ", ".join(
        [str(index) for indexValueReferencePair in inputs for index in indexValueReferencePair]
    )
    flattened_output_string = ", ".join(
        [str(index) for indexValueReferencePair in outputs for index in indexValueReferencePair]
    )
    flattened_state_string = ", ".join([str(index) for index in state_output_indexes])
    flattened_state_init_string = ", ".join(
        [str(index) for indexValueReferencePair in state_init for index in indexValueReferencePair]
    )

    template_data: dict[str, str] = {
        "numFmuVariables": str(num_fmu_variables),
        "FmuName": fmi_model.name,
        "numOnnxInputs": str(onnx.input_size),
        "numOnnxOutputs": str(onnx.output_size),
        "numOnnxStates": str(onnx.state_size),
        "numOnnxStateInit": str(num_onnx_state_init),
        "onnxUsesTime": "true" if onnx.time_input else "false",
        "onnxInputName": onnx.input_name,
        "onnxStatesName": onnx.states_name,
        "onnxTimeInputName": onnx.time_input_name,
        "onnxOutputName": onnx.output_name,
        "onnxFileName": onnx.filename,
        "numOnnxFmuInputs": str(num_fmu_inputs),
        "numOnnxFmuOutputs": str(num_fmu_outputs),
        "numOnnxStatesOutputs": str(num_onnx_states),
        "onnxInputValueReferences": flattened_input_string,
        "onnxOutputValueReferences": flattened_output_string,
        "onnxStateOutputIndexes": flattened_state_string,
        "onnxStateInitValueReferences": flattened_state_init_string,
    }

    return template_data


def validate_interface_spec(
    spec: str,
) -> tuple[ValidationError | None, ModelComponent]:
    """
    Parse and validate JSON data from the interface file.

    Args
    ----
        spec (str): Contents of the JSON file.

    Returns
    -------
        tuple[Optional[ValidationError], ModelComponent]:
            The validation error (if any) and the validated model component.
            The pydantic model instance that contains all the interface information.
    """
    parsed_spec = ModelComponent.model_validate_json(json_data=spec, strict=True)
    try:
        validated_model = ModelComponent.model_validate(parsed_spec)
    except ValidationError as e:
        return e, parsed_spec

    return None, validated_model


def generate_fmu_files(
    fmu_src_path: os.PathLike[str],
    onnx_path: os.PathLike[str],
    interface_spec_path: os.PathLike[str],
) -> FmiModel:
    """
    Generate FMU files based on the FMU source, ONNX model, and interface specification.

    Args
    ----
        fmu_src_path (os.PathLike[str]): The path to the FMU source directory.
        onnx_path (os.PathLike[str]): The path to the ONNX model file.
        interface_spec_path (os.PathLike[str]): The path to the interface specification file.

    Returns
    -------
        FmiModel: The FMI model.
    """
    # Create Path instances for the path to the spec and ONNX file.
    onnx_path = Path(onnx_path)
    interface_spec_path = Path(interface_spec_path)

    # Load JSON interface contents
    with Path.open(interface_spec_path, encoding="utf-8") as template_file:
        interface_contents = template_file.read()

    # Validate the FMU interface spec against expected Schema
    error, component_model = validate_interface_spec(interface_contents)

    if error:
        # Display error and finish workflow
        raise error

    # Create ONNXModel and FmiModel instances -> load some metadata
    onnx_model = ONNXModel(onnx_path=onnx_path, time_input=bool(component_model.uses_time))
    fmi_model = FmiModel(model=component_model)
    fmu_source = Path(fmu_src_path) / fmi_model.name

    template_data = format_template_data(onnx=onnx_model, fmi_model=fmi_model, model_component=component_model)

    # Generate all FMU files
    make_fmu_dirs(fmu_source)
    create_files_from_templates(data=template_data, fmu_src=fmu_source)
    create_model_description(fmu=fmi_model, src_path=fmu_source)

    # Copy ONNX file and save it inside FMU folder
    _ = shutil.copyfile(src=onnx_path, dst=fmu_source / "resources" / onnx_model.filename)

    return fmi_model


def validate_fmu_source_files(fmu_path: os.PathLike[str]) -> None:
    """
    Validate the FMU source files.

    Args
    ----
        fmu_path (os.PathLike[str]): The path to the FMU source directory.

    Raises
    ------
        FileNotFoundError: If required files are missing in the FMU source directory.
    """
    fmu_path = Path(fmu_path)

    files_should_exist: list[str] = [
        "modelDescription.xml",
        "sources/fmu.cpp",
        "sources/model_definitions.h",
    ]

    if files_not_exists := [file for file in files_should_exist if not (fmu_path / file).is_file()]:
        raise FileNotFoundError(
            f"The files {files_not_exists} are not contained in the provided FMU source path ({fmu_path})"
        )

    resources_dir = fmu_path / "resources"

    num_onnx_files = len(list(resources_dir.glob("*.onnx")))

    if num_onnx_files < 1:
        raise FileNotFoundError(
            f"There is no *.onnx file in the resource folder in the provided FMU source path ({fmu_path})"
        )


def build_fmu(
    fmu_src_path: os.PathLike[str],
    fmu_build_path: os.PathLike[str],
    fmu_save_path: os.PathLike[str],
) -> None:
    """
    Build the FMU.

    Args
    ----
        fmu_src_path (os.PathLike[str]): The path to the FMU source directory.
        fmu_build_path (os.PathLike[str]): The path to the FMU build directory.
        fmu_save_path (os.PathLike[str]): The path to save the built FMU.

    Raises
    ------
        FileNotFoundError: If required files are missing in the FMU source directory.
    """
    fmu_src_path = Path(fmu_src_path)
    validate_fmu_source_files(fmu_src_path)
    fmu_name = fmu_src_path.stem
    conan_install_command = [
        "conan",
        "install",
        ".",
        "-of",
        str(fmu_build_path),
        "-u",
        "-b",
        "missing",
        "-o",
        "shared=True",
    ]
    cmake_set_folders = [
        f"-DCMAKE_BINARY_DIR={fmu_build_path!s}",
        f"-DFMU_OUTPUT_DIR={fmu_save_path!s}",
        f"-DFMU_NAMES={fmu_name}",
        f"-DFMU_SOURCE_PATH={fmu_src_path.parent!s}",
    ]
    # Windows vs Linux
    conan_preset = "conan-default" if os.name == "nt" else "conan-release"
    cmake_command = ["cmake", *cmake_set_folders, "--preset", conan_preset]
    cmake_build_command = ["cmake", "--build", ".", "-j", "14", "--config", "Release"]

    # Change directory to the build folder
    os.chdir(fmu_build_folder)

    # Run conan install, cmake, cmake build
    logger.debug("Builder: Run conan install")
    try:
        _ = subprocess.run(conan_install_command, check=True)  # noqa: S603
    except subprocess.CalledProcessError:
        logger.exception("Exception in conan install: %s")

    logger.debug("Builder: Run cmake")
    try:
        _ = subprocess.run(cmake_command, check=True)  # noqa: S603
    except subprocess.CalledProcessError:
        logger.exception("Exception in cmake: %s")

    os.chdir(fmu_build_path)
    logger.debug("Builder: Run cmake build")
    try:
        _ = subprocess.run(cmake_build_command, check=True)  # noqa: S603
    except subprocess.CalledProcessError:
        logger.exception("Exception in cmake build: %s")

    logger.debug("Builder: Done with build_fmu")

    # Return to original working directory (leave build dir)
    os.chdir(absolute_path)
