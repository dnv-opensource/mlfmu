import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from mlfmu.types.fmu_component import FmiModel, ModelComponent
from mlfmu.types.onnx_model import ONNXModel
from mlfmu.utils.fmi_builder import generate_model_description
from mlfmu.utils.signals import range_list_expanded

# Hard coded values for testing functionality
absolute_path = Path().absolute()
# TODO: I had some problems with this absolute_path.parent.parent, so I changed it to this to make it work.
# These are just temporary hard coded values that should be provided by the user. So it isn't that important.
template_parent_path = absolute_path / "templates" / "fmu"
json_interface = absolute_path / "examples" / "wind_generator" / "config" / "interface.json"
fmu_src_path = absolute_path / "examples" / "wind_generator"
onnx_path = absolute_path / "examples" / "wind_generator" / "config" / "example.onnx"
build_path = absolute_path / "build_fmu"
save_fmu_path = absolute_path / "fmus"


# Replacing all the template strings with their corresponding values and saving to new file
def format_template_file(template_path: Path, save_path: Path, data: dict[str, str]):
    # TODO: Need to check that these calls are safe from a cybersecurity point of view
    with open(template_path, "r", encoding="utf-8") as template_file:
        template_string = template_file.read()

    formatted_string = template_string.format(**data)
    with open(save_path, "w", encoding="utf-8") as save_file:
        _ = save_file.write(formatted_string)


def create_model_description(fmu: FmiModel, src_path: Path):
    # Compute XML structure for FMU
    xml_structure = generate_model_description(fmu_model=fmu)

    # Save in file
    xml_structure.write(src_path / "modelDescription.xml", encoding="utf-8")


# Creating all the directories needed to put all the FMU files in
def make_fmu_dirs(src_path: Path):
    sources_path = src_path / "sources"
    resources_path = src_path / "resources"
    sources_path.mkdir(parents=True, exist_ok=True)
    resources_path.mkdir(parents=True, exist_ok=True)


# Creating and formatting all needed c++ files for FMU generation
def create_files_from_templates(data: dict[str, str], fmu_src: Path):
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


# Function for generating the key value pairs needed to format the template files to valid c++
def format_template_data(onnx: ONNXModel, fmi_model: FmiModel, model_component: ModelComponent) -> dict[str, str]:
    # Work out template mapping between ONNX and FMU ports
    # TODO: Get information about the parameters for initalizing state and add that info to the template
    # Initialization indexes should be formatted as onnxInputValueReferences: state_index, value_reference, state_index, value_reference, ...
    inputs, outputs = fmi_model.get_template_mapping()
    state_output_indexes = [
        index for state in model_component.states for index in range_list_expanded(state.agent_output_indexes)
    ]

    # Total number of inputs/outputs/internal states
    num_fmu_inputs = len(inputs)
    num_fmu_outputs = len(outputs)
    num_onnx_states = len(state_output_indexes)

    # Checking compatibility between ModelComponent and ONNXModel
    if num_fmu_inputs > onnx.input_size:
        raise ValueError(
            f"The number of total input indexes for all inputs and parameter in the interface file(={num_fmu_inputs}) cannot exceed the input size of the ml model (={onnx.input_size})"
        )
    if num_fmu_outputs > onnx.output_size:
        raise ValueError(
            f"The number of total output indexes for all outputs in the interface file(={num_fmu_outputs}) cannot exceed the output size of the ml model (={onnx.output_size})"
        )
    if num_onnx_states > min(onnx.state_size, onnx.output_size):
        raise ValueError(
            f"The number of total output indexes for all states in the interface file(={num_onnx_states}) cannot exceed either the state input size (={onnx.state_size}) or the output size of the ml model (={onnx.output_size})"
        )

    # Flatten vectors to comply with template requirements -> onnx-index, variable-reference, onnx-index, variable-reference ...
    flattened_input_string = ", ".join(
        [str(index) for indexValueReferencePair in inputs for index in indexValueReferencePair]
    )
    flattened_output_string = ", ".join(
        [str(index) for indexValueReferencePair in outputs for index in indexValueReferencePair]
    )
    flattened_state_string = ", ".join([str(index) for index in state_output_indexes])

    template_data: dict[str, str] = dict(
        numFmuVariables=str(fmi_model.get_total_variable_number()),
        FmuName=fmi_model.name,
        numOnnxInputs=str(onnx.input_size),
        numOnnxOutputs=str(onnx.output_size),
        numOnnxStates=str(onnx.state_size),
        onnxUsesTime="true" if onnx.time_input else "false",
        onnxInputName=onnx.input_name,
        onnxStatesName=onnx.states_name,
        onnxTimeInputName=onnx.time_input_name,
        onnxOutputName=onnx.output_name,
        onnxFileName=onnx.filename,
        numOnnxFmuInputs=str(num_fmu_inputs),
        numOnnxFmuOutputs=str(num_fmu_outputs),
        numOnnxStatesOutputs=str(num_onnx_states),
        onnxInputValueReferences=flattened_input_string,
        onnxOutputValueReferences=flattened_output_string,
        onnxStateOutputIndexes=flattened_state_string,
        numOnnxStateInit="0",
        onnxStateInitValueReferences="",
    )

    return template_data


def validate_interface_spec(
    spec: str,
) -> tuple[Optional[ValidationError], ModelComponent]:
    """Parse and validate JSON data from interface file.

    Args:
        spec (str): Contents of JSON file.

    Returns
    -------
        The pydantic model instance that contains all the interface information.
    """
    parsed_spec = ModelComponent.model_validate_json(json_data=spec, strict=True)

    try:
        validated_model = ModelComponent.model_validate(parsed_spec)
    except ValidationError as e:
        return e, parsed_spec

    return None, validated_model


def generate_fmu_files(
    fmu_src_path: os.PathLike[str], onnx_path: os.PathLike[str], interface_spec_path: os.PathLike[str]
):
    # Create Path instances for the path to the spec and ONNX file.
    onnx_path = Path(onnx_path)
    interface_spec_path = Path(interface_spec_path)

    # Load JSON interface contents
    with open(interface_spec_path, "r", encoding="utf-8") as template_file:
        interface_contents = template_file.read()

    # Validate the FMU interface spec against expected Schema
    error, component_model = validate_interface_spec(interface_contents)

    if error:
        # Display error and finish workflow
        print(error)
        return

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


def validate_fmu_source_files(fmu_path: os.PathLike[str]):
    fmu_path = Path(fmu_path)

    files_should_exist: List[str] = [
        "modelDescription.xml",
        "sources/fmu.cpp",
        "sources/model_definitions.h",
    ]

    files_not_exists = [file for file in files_should_exist if not (fmu_path / file).is_file()]

    if len(files_not_exists) > 0:
        raise FileNotFoundError(
            f"The files {files_not_exists} are not contained in the provided fmu source path ({fmu_path})"
        )

    resources_dir = fmu_path / "resources"

    num_onnx_files = len(list(resources_dir.glob("*.onnx")))

    if num_onnx_files < 1:
        raise FileNotFoundError(
            f"There is no *.onnx file in the resource folder in the provided fmu source path ({fmu_path})"
        )


def build_fmu(
    fmi_model: FmiModel,
    fmu_src_path: os.PathLike[str],
    fmu_build_path: os.PathLike[str],
    fmu_save_path: os.PathLike[str],
):
    validate_fmu_source_files(Path(fmu_src_path) / fmi_model.name)

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
        f"-DCMAKE_BINARY_DIR={str(fmu_build_path)}",
        f"-DFMU_OUTPUT_DIR={str(fmu_save_path)}",
        f"-DFMU_NAMES={fmi_model.name}",
        f"-DFMU_SOURCE_PATH={str(fmu_src_path)}",
    ]

    cmake_command = ["cmake", *cmake_set_folders, "--preset", "conan-default"]

    cmake_build_command = ["cmake", "--build", ".", "-j", "14", "--config", "Release"]

    _ = subprocess.run(conan_install_command)
    _ = subprocess.run(cmake_command)
    os.chdir(fmu_build_path)
    _ = subprocess.run(cmake_build_command)
    os.chdir(os.getcwd())

    # TODO: Clean up.

    pass


if __name__ == "__main__":
    fmi_model = generate_fmu_files(fmu_src_path=fmu_src_path, onnx_path=onnx_path, interface_spec_path=json_interface)
    if fmi_model is None:
        exit()
    build_fmu(fmi_model, fmu_src_path, build_path, save_fmu_path)
