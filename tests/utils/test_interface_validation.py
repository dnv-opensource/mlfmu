import json

from pydantic import ValidationError

from mlfmu.types.fmu_component import ModelComponent
from mlfmu.utils.builder import validate_interface_spec


def test_validate_simple_interface_spec():
    # Assuming validate_interface_spec takes a dictionary as input
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [{"name": "input1", "description": "My input1", "agentInputIndexes": ["0"], "type": "integer"}],
        "outputs": [{"name": "output1", "description": "My output1", "agentInputIndexes": ["0"]}],
    }
    error, model = validate_interface_spec(json.dumps(valid_spec))
    assert error is None
    assert isinstance(model, ModelComponent)
    assert model.name == "example"
    assert model.version == "1.0"
    assert model.inputs[0].name == "input1"
    assert model.inputs[0].type == "integer"
    assert model.outputs[0].name == "output1"
    assert model.outputs[0].type == "real"


def test_validate_interface_spec_wrong_types():
    # Assuming validate_interface_spec returns False for invalid specs
    invalid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [{"name": "input1", "type": "enum"}],  # Missing enum type
        "outputs": [{"name": "output1", "type": "int"}],  # Should be integer
    }
    error, model = validate_interface_spec(json.dumps(invalid_spec))
    assert error
    assert model is None
    assert isinstance(error, ValidationError)
    # Model type error as it's missing the agentInputIndexes
    assert error.errors()[0]["msg"] == "Input should be 'real', 'integer', 'string' or 'boolean'"
    assert error.errors()[1]["msg"] == "Input should be 'real', 'integer', 'string' or 'boolean'"


def test_validate_unnamed_spec():
    invalid_spec = {
        "version": "1.0",
        "inputs": [{"name": "input1", "description": "My input1", "agentInputIndexes": ["0"], "type": "integer"}],
        "outputs": [{"name": "output1", "description": "My output1", "agentInputIndexes": ["0"]}],
    }
    error, model = validate_interface_spec(json.dumps(invalid_spec))
    assert error
    assert model is None
    assert isinstance(error, ValidationError)
    assert error.errors()[0]["msg"] == "Field required"


def test_validate_invalid_agent_indices():
    invalid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "input1", "description": "My input1", "type": "integer", "agentInputIndexes": [0, ":10", "10.0"]}
        ],  # Should be a stringified list of integers
        "outputs": [
            {"name": "output1", "description": "My output1", "agentOutputIndexes": ["0:a"]}
        ],  # Should not have letters
    }
    error, model = validate_interface_spec(json.dumps(invalid_spec))
    assert error
    assert model is None
    assert isinstance(error, ValidationError)
    assert len(error.errors()) == 4
    assert error.errors()[0]["msg"] == "Input should be a valid string"
    assert error.errors()[1]["msg"] == "String should match pattern '^(\\d+|\\d+:\\d+)$'"
    assert error.errors()[1]["msg"] == "String should match pattern '^(\\d+|\\d+:\\d+)$'"
    assert error.errors()[2]["msg"] == "String should match pattern '^(\\d+|\\d+:\\d+)$'"


def test_validate_default_parameters():
    invalid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "input1", "description": "My input1", "type": "integer", "agentInputIndexes": ["10"]}
        ],  # Should be a stringified list of integers
        "outputs": [
            {"name": "output1", "description": "My output1", "agentOutputIndexes": ["0:10"]}
        ],  # Should not have letters
    }
    error, model = validate_interface_spec(json.dumps(invalid_spec))
    assert error is None
    assert model is not None

    assert model.uses_time is False
    assert model.state_initialization_reuse is False
    assert model.name == "example"
    assert model.parameters == []


def test_validate_internal_states():
    invalid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {"name": "input1", "description": "My input1", "type": "integer", "agentInputIndexes": ["10"]}
        ],  # Should be a stringified list of integers
        "outputs": [
            {"name": "output1", "description": "My output1", "agentOutputIndexes": ["0:10"]}
        ],  # Should not have letters
        "states": [
            {
                "name": "state1",
                "description": "My state1",
                "startValue": 10,
                "initializationVariable": "input1",
                "agentOutputIndexes": ["0:10"],
            },
            {"name": "state2", "description": "My state2", "agentOutputIndexes": ["0:10"]},
            {"name": "state3", "initializationVariable": "input1", "agentOutputIndexes": ["0:10"]},
            {"description": "My state4", "startValue": 10},
        ],
    }
    error, model = validate_interface_spec(json.dumps(invalid_spec))
    assert error is not None
    assert model is None
    assert isinstance(error, ValidationError)
    assert (
        error.errors()[0]["msg"]
        == "Value error, Only one state initialization method is allowed to be used at a time: initialization_variable cannot be set if either start_value or name is set."
    )
    assert (
        error.errors()[1]["msg"]
        == "Value error, name is set without start_value being set. Both fields need to be set for the state initialization to be valid"
    )
    assert (
        error.errors()[2]["msg"]
        == "Value error, Only one state initialization method is allowed to be used at a time: initialization_variable cannot be set if either start_value or name is set."
    )
    assert (
        error.errors()[3]["msg"]
        == "Value error, start_value is set without name being set. Both fields need to be set for the state initialization to be valid"
    )
