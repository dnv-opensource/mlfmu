import logging
import datetime
import pkg_resources
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree, SubElement
from mlfmu.types.FMU_component import FmiCausality, FmiModel, FmiVariability, FmiVariable, ModelComponent

logger = logging.getLogger(__name__)

def requires_start(var: FmiVariable) -> bool:
    """Test if a variable requires a start attribute

    Returns:
        True if successful, False otherwise
    """
    return (
        var.causality == FmiCausality.INPUT
        or var.causality == FmiCausality.PARAMETER
        or var.variability == FmiVariability.CONSTANT
    )


def generate_model_description(fmu_component: FmiModel) -> ElementTree:
    """Generate FMU modelDescription as XML.

    Args:
        fmu_component (FmiModel): Object representation of FMI slave instance

    returns:
        xml.etree.TreeElement.Element: modelDescription XML representation.
    """

    t = datetime.datetime.now(datetime.timezone.utc)
    date_str = t.isoformat(timespec="seconds")
    TOOL_VERSION = pkg_resources.get_distribution("MLFMU").version

    # Root <fmiModelDescription> tag
    model_description = dict(
        fmiVersion="2.0",
        modelName=fmu_component.name,
        guid=f"{fmu_component.guid!s}" if fmu_component.guid is not None else "@FMU_UUID@",
        version=fmu_component.version,
        generationDateAndTime=date_str,
        variableNamingConvention="structured",
        generationTool=f"MLFMU {TOOL_VERSION}",
    )

    # Optional props
    if fmu_component.copyright is not None:
        model_description["copyright"] = fmu_component.copyright
    if fmu_component.license is not None:
        model_description["license"] = fmu_component.license
    if fmu_component.author is not None:
        model_description["author"] = fmu_component.author
    if fmu_component.description is not None:
        model_description["description"] = fmu_component.description

    root = Element("fmiModelDescription", model_description)

    # <CoSimulation> tag options
    cosim_options = dict(modelIdentifier=fmu_component.name, canHandleVariableCommunicationStepSize="true")
    _ = SubElement(root, "CoSimulation", attrib=cosim_options)

    # <ModelVariables> tag -> Append inputs/parameters/outputs
    variables = SubElement(root, "ModelVariables")
    for var in fmu_component.get_fmi_model_variables():
        # XML variable attributes
        var_attrs = dict(
            name=var.name,
            valueReference=str(var.variable_reference),
            causality=var.causality.value,
            description=var.description if var.description else "",
            variability=var.variability.value if var.variability else FmiVariability.CONTINUOUS.value,
        )
        var_elem = SubElement(variables, "ScalarVariable", var_attrs)

        var_type_attrs = dict()
        if requires_start(var):
            var_type_attrs["start"] = str(var.start_value)

        # FMI variable type element
        _ = SubElement(var_elem, var.type.value.capitalize(), var_type_attrs)

    # Create XML tree containing root element and pretty format its contents
    xml_tree = ElementTree(root)
    ET.indent(xml_tree, space="\t", level=0)
    return xml_tree
