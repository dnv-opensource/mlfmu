import json
import re
from pathlib import Path

import pytest

from mlfmu.types.fmu_component import FmiModel
from mlfmu.types.onnx_model import ONNXModel
from mlfmu.utils.builder import format_template_data, validate_interface_spec


@pytest.fixture(scope="session")
def wind_generator_onnx() -> ONNXModel:
    return ONNXModel(Path.cwd().parent / "data" / "example.onnx", time_input=True)


def test_valid_template_data(wind_generator_onnx: ONNXModel):
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "inputs", "description": "My inputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2}
        ],
        "outputs": [
            {
                "name": "outputs",
                "description": "My outputs",
                "agentOutputIndexes": ["0:2"],
                "isArray": True,
                "length": 2,
            }
        ],
        "states": [
            {"agentOutputIndexes": ["2:130"]},
            {"name": "state1", "startValue": 10.0, "agentOutputIndexes": ["0"]},
            {"name": "state2", "startValue": 180.0, "agentOutputIndexes": ["1"]},
        ],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    template_data = format_template_data(onnx=wind_generator_onnx, fmi_model=fmi_model, model_component=model)

    assert template_data["FmuName"] == "example"
    assert template_data["numFmuVariables"] == "6"
    assert template_data["numOnnxInputs"] == "2"
    assert template_data["numOnnxOutputs"] == "130"
    assert template_data["numOnnxStates"] == "130"
    assert template_data["onnxInputValueReferences"] == "0, 0, 1, 1"
    assert template_data["onnxOutputValueReferences"] == "0, 2, 1, 3"


def test_template_data_invalid_input_size(wind_generator_onnx: ONNXModel):
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "inputs", "description": "My inputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2},
            {
                "name": "inputs2",
                "description": "My inputs 2",
                "agentInputIndexes": ["0:10"],
                "isArray": True,
                "length": 10,
            },
        ],
        "outputs": [
            {"name": "outputs", "description": "My outputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2}
        ],
        "states": [
            {"agentOutputIndexes": ["2:130"]},
            {"name": "state1", "startValue": 10.0, "agentOutputIndexes": ["0"]},
            {"name": "state2", "startValue": 180.0, "agentOutputIndexes": ["1"]},
        ],
    }

    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)

    with pytest.raises(ValueError) as exc_info:
        _ = format_template_data(onnx=wind_generator_onnx, fmi_model=fmi_model, model_component=model)

    assert exc_info.match(
        re.escape(
            "The number of total input indexes for all inputs and parameter in the interface file (=12) \
cannot exceed the input size of the ml model (=2)"
        )
    )


def test_template_data_invalid_output_size(wind_generator_onnx: ONNXModel):
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "inputs", "description": "My inputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2}
        ],
        "outputs": [
            {
                "name": "outputs",
                "description": "My outputs",
                "agentOutputIndexes": ["0:2"],
                "isArray": True,
                "length": 2,
            },
            {
                "name": "outputs2",
                "description": "My outputs 2",
                "agentOutputIndexes": ["0:200"],
                "isArray": True,
                "length": 200,
            },
        ],
        "states": [
            {"agentOutputIndexes": ["2:130"]},
            {"name": "state1", "startValue": 10.0, "agentOutputIndexes": ["0"]},
            {"name": "state2", "startValue": 180.0, "agentOutputIndexes": ["1"]},
        ],
    }

    _, model = validate_interface_spec(json.dumps(valid_spec))
    fmi_model = FmiModel(model=model)

    with pytest.raises(ValueError) as exc_info:
        _ = format_template_data(onnx=wind_generator_onnx, fmi_model=fmi_model, model_component=model)

    assert exc_info.match(
        re.escape(
            "The number of total output indexes for all outputs in the interface file (=202) \
cannot exceed the output size of the ml model (=130)"
        )
    )


def test_template_data_invalid_state_size(wind_generator_onnx: ONNXModel):
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "inputs", "description": "My inputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2}
        ],
        "outputs": [
            {"name": "outputs", "description": "My outputs", "agentInputIndexes": ["0:2"], "isArray": True, "length": 2}
        ],
        "states": [
            {"agentOutputIndexes": ["2:200"]},
        ],
    }

    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)

    with pytest.raises(ValueError) as exc_info:
        _ = format_template_data(onnx=wind_generator_onnx, fmi_model=fmi_model, model_component=model)

    assert exc_info.match(
        re.escape(
            "The number of total output indexes for all states in the interface file (=198) \
cannot exceed either the state input size (=130)"
        )
    )
