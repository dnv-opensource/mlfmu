import datetime
import logging
from importlib import metadata
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from mlfmu.types.fmu_component import (
    FmiCausality,
    FmiModel,
    FmiVariability,
    FmiVariable,
)

logger = logging.getLogger(__name__)


def requires_start(var: FmiVariable) -> bool:
    """
    Test if a variable requires a start attribute.

    Returns
    -------
        True if successful, False otherwise
    """
    return var.causality in (FmiCausality.INPUT, FmiCausality.PARAMETER) or var.variability == FmiVariability.CONSTANT


def generate_model_description(fmu_model: FmiModel) -> ElementTree:
    """
    Generate FMU modelDescription as XML.

    Args
    ----
        fmu_model (FmiModel): Object representation of FMI slave instance

    Returns
    -------
        xml.etree.TreeElement.Element: modelDescription XML representation.
    """
    t = datetime.datetime.now(datetime.timezone.utc)
    date_str = t.isoformat(timespec="seconds")
    TOOL_VERSION = metadata.version("mlfmu")  # noqa: N806

    # Root <fmiModelDescription> tag
    model_description = {
        "fmiVersion": "2.0",
        "modelName": fmu_model.name,
        "guid": f"{fmu_model.guid!s}" if fmu_model.guid is not None else "@FMU_UUID@",
        "version": fmu_model.version,
        "generationDateAndTime": date_str,
        "variableNamingConvention": "structured",
        "generationTool": f"MLFMU {TOOL_VERSION}",
    }

    # Optional props
    if fmu_model.copyright is not None:
        model_description["copyright"] = fmu_model.copyright
    if fmu_model.license is not None:
        model_description["license"] = fmu_model.license
    if fmu_model.author is not None:
        model_description["author"] = fmu_model.author
    if fmu_model.description is not None:
        model_description["description"] = fmu_model.description

    root = Element("fmiModelDescription", model_description)

    # <CoSimulation> tag options
    cosim_options = {
        "modelIdentifier": fmu_model.name,
        "canHandleVariableCommunicationStepSize": "true",
    }
    _ = SubElement(root, "CoSimulation", attrib=cosim_options)

    # <ModelVariables> tag -> Append inputs/parameters/outputs
    variables = SubElement(root, "ModelVariables")

    # <ModelStructure> tag with <Outputs> tab inside --> Append all outputs
    model_structure = SubElement(root, "ModelStructure")
    outputs = SubElement(model_structure, "Outputs")
    initial_unknowns = SubElement(model_structure, "InitialUnknowns")

    # Get all variables to add them inside the <ModelVariables> tag
    model_variables = fmu_model.get_fmi_model_variables()

    # The variables needs to be added in the order of their valueReference
    sorted_model_variables = sorted(model_variables, key=lambda x: x.variable_reference)

    # Add each variable inside the <ModelVariables> tag
    for var in sorted_model_variables:
        # XML variable attributes
        var_attrs = {
            "name": var.name,
            "valueReference": str(var.variable_reference),
            "causality": var.causality.value,
            "description": var.description or "",
            "variability": var.variability.value if var.variability else FmiVariability.CONTINUOUS.value,
        }
        var_elem = SubElement(variables, "ScalarVariable", var_attrs)

        var_type_attrs = {}
        if requires_start(var):
            var_type_attrs["start"] = str(var.start_value)

        # FMI variable type element
        _ = SubElement(var_elem, var.type.value.capitalize(), var_type_attrs)

        # Adding outputs inside <ModelStructure>
        if var.causality == FmiCausality.OUTPUT:
            # Index is 1-indexed for <Unknown> tag
            unknown_attributes = {"index": str(var.variable_reference + 1)}
            # For each output create an <Unknown> tag inside both <Outputs> and <InitialUnknowns>
            _ = SubElement(outputs, "Unknown", unknown_attributes)
            _ = SubElement(initial_unknowns, "Unknown", unknown_attributes)

    # Create XML tree containing root element and pretty format its contents
    xml_tree = ElementTree(root)
    indent(xml_tree, space="\t", level=0)
    return xml_tree
