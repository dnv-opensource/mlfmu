import json

from mlfmu.types.fmu_component import FmiModel
from mlfmu.utils.builder import validate_interface_spec
from mlfmu.utils.fmi_builder import generate_model_description


def test_generate_simple_model_description():
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [{"name": "input1", "description": "My input1", "agentInputIndexes": ["0"], "type": "integer"}],
        "outputs": [{"name": "output1", "description": "My output1", "agentInputIndexes": ["0"]}],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    xml_structure = generate_model_description(fmu_model=fmi_model)
    variables = xml_structure.findall(".//ScalarVariable")

    root = xml_structure.getroot()
    assert root is not None, "XML structure has no root element"
    assert root.tag == "fmiModelDescription"
    assert variables[0].attrib["name"] == "input1"
    assert variables[0].attrib["causality"] == "input"
    assert variables[0].attrib["variability"] == "continuous"
    assert variables[0].attrib["description"] == "My input1"
    assert variables[0][0].tag == "Integer"

    assert variables[1].attrib["name"] == "output1"
    assert variables[1].attrib["causality"] == "output"
    assert variables[1].attrib["variability"] == "continuous"
    assert variables[1].attrib["description"] == "My output1"
    assert variables[1][0].tag == "Real"


def test_generate_model_description_with_internal_state_params():
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "states": [
            {
                "name": "state1",
                "description": "My state1",
                "startValue": 0.0,
                "type": "real",
                "agentOutputIndexes": ["0"],
            }
        ],
        "outputs": [{"name": "output1", "description": "My output1", "agentInputIndexes": ["0"]}],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    xml_structure = generate_model_description(fmu_model=fmi_model)
    variables = xml_structure.findall(".//ScalarVariable")

    root = xml_structure.getroot()
    assert root is not None, "XML structure has no root element"
    assert root.tag == "fmiModelDescription"

    assert variables[0].attrib["name"] == "output1"
    assert variables[0].attrib["causality"] == "output"

    assert variables[1].attrib["name"] == "state1"
    assert variables[1].attrib["causality"] == "parameter"
    assert variables[1][0].tag == "Real"
    assert variables[1][0].attrib["start"] == "0.0"


def test_generate_vector_ports():
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "inputs": [
            {
                "name": "inputVector",
                "description": "My input1",
                "agentInputIndexes": ["0:5"],
                "type": "real",
                "isArray": True,
                "length": 5,
            }
        ],
        "outputs": [
            {
                "name": "outputVector",
                "description": "My output1",
                "agentInputIndexes": ["0:5"],
                "isArray": True,
                "length": 5,
            }
        ],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    xml_structure = generate_model_description(fmu_model=fmi_model)
    variables = xml_structure.findall(".//ScalarVariable")

    assert model
    assert variables[0].attrib["name"] == "inputVector[0]"
    assert variables[1].attrib["name"] == "inputVector[1]"
    assert variables[2].attrib["name"] == "inputVector[2]"
    assert variables[3].attrib["name"] == "inputVector[3]"
    assert variables[4].attrib["name"] == "inputVector[4]"

    assert variables[5].attrib["name"] == "outputVector[0]"
    assert variables[6].attrib["name"] == "outputVector[1]"
    assert variables[7].attrib["name"] == "outputVector[2]"
    assert variables[8].attrib["name"] == "outputVector[3]"
    assert variables[9].attrib["name"] == "outputVector[4]"


def test_generate_model_description_with_start_value():
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "usesTime": True,
        "inputs": [
            {
                "name": "input1",
                "description": "My input1",
                "agentInputIndexes": ["0"],
                "type": "integer",
                "startValue": 10,
            },
            {
                "name": "input2",
                "description": "My input2",
                "agentOutputIndexes": ["0"],
                "type": "boolean",
                "startValue": True,
            },
            {"name": "input3", "description": "My input3", "agentOutputIndexes": ["0"], "startValue": 10.0},
        ],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    xml_structure = generate_model_description(fmu_model=fmi_model)
    variables = xml_structure.findall(".//ScalarVariable")

    root = xml_structure.getroot()
    assert root is not None, "XML structure has no root element"
    assert root.tag == "fmiModelDescription"
    assert variables[0].attrib["name"] == "input1"
    assert variables[0].attrib["causality"] == "input"
    assert variables[0][0].tag == "Integer"
    assert variables[0][0].attrib["start"] == "10"

    assert variables[1].attrib["name"] == "input2"
    assert variables[1].attrib["causality"] == "input"
    assert variables[1][0].tag == "Boolean"
    assert variables[1][0].attrib["start"] == "True"

    assert variables[2].attrib["name"] == "input3"
    assert variables[2].attrib["causality"] == "input"
    assert variables[2][0].tag == "Real"
    assert variables[2][0].attrib["start"] == "10.0"


def test_generate_model_description_output():
    valid_spec = {
        "name": "example",
        "version": "1.0",
        "usesTime": True,
        "inputs": [
            {
                "name": "input1",
                "description": "My input1",
                "agentInputIndexes": ["0"],
                "type": "integer",
                "startValue": 10,
            },
            {
                "name": "input2",
                "description": "My input2",
                "agentOutputIndexes": ["0"],
                "type": "boolean",
                "startValue": True,
            },
            {"name": "input3", "description": "My input3", "agentOutputIndexes": ["0"], "startValue": 10.0},
        ],
        "outputs": [
            {"name": "output1", "description": "My output1", "agentInputIndexes": ["0"], "type": "real"},
            {"name": "output1", "description": "My output1", "agentInputIndexes": ["0"], "type": "real"},
        ],
    }
    _, model = validate_interface_spec(json.dumps(valid_spec))
    assert model is not None

    fmi_model = FmiModel(model=model)
    xml_structure = generate_model_description(fmu_model=fmi_model)
    variables = xml_structure.findall(".//ScalarVariable")
    output_variables = [var for var in variables if var.attrib.get("causality") == "output"]
    outputs_registered = xml_structure.findall(".//Outputs/Unknown")

    # The index should be the valueReference + 1
    assert int(output_variables[0].attrib["valueReference"]) + 1 == int(outputs_registered[0].attrib["index"])
    assert int(output_variables[1].attrib["valueReference"]) + 1 == int(outputs_registered[1].attrib["index"])
