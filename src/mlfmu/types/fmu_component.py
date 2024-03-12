from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, root_validator
from pydantic.fields import Field
from typing_extensions import Annotated

from mlfmu.types.component_examples import create_fmu_signal_example
from mlfmu.utils.signals import range_list_expanded
from mlfmu.utils.strings import to_camel


class FmiVariableType(str, Enum):
    """Enum for variable type."""

    REAL = "real"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"


class FmiCausality(str, Enum):
    """Enum for variable causality."""

    PARAMETER = "parameter"
    INPUT = "input"
    OUTPUT = "output"


class FmiVariability(str, Enum):
    """Enum for signal variability."""

    CONSTANT = "constant"
    FIXED = "fixed"
    TUNABLE = "tunable"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class BaseModelConfig(BaseModel):
    """Enables the alias_generator for all cases."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Variable(BaseModelConfig):
    name: str = Field(
        None,
        description="Unique name for the port.",
        examples=["windSpeed", "windDirection"],
    )
    type: Optional[FmiVariableType] = Field(
        FmiVariableType.REAL,
        description="Data type as defined by FMI standard, defaults to Real.",
        examples=[FmiVariableType.REAL, FmiVariableType.INTEGER],
    )
    description: Optional[str] = Field(None, description="Short FMU variable description.")
    variability: Optional[FmiVariability] = Field(None, description="Signal variability as defined by FMI.")
    start_value: Optional[Union[float, str, bool, int]] = Field(
        0,
        description="Initial value of the signal at time step 1. Type should match the variable type.",
    )
    is_array: Optional[bool] = Field(
        False,
        description="When dealing with an array signal, it is essential to specify the LENGTH parameter. Arrays are indexed starting from 0, and FMU signals will be structured as SIGNAL_NAME[0], SIGNAL_NAME[1], and so forth. By default, this feature is set to False.",
    )
    length: Optional[int] = Field(
        None,
        description="Defines the number of entries in the signal if the signal is array.",
        examples=[3, 5],
    )


class InternalState(BaseModelConfig):
    name: Optional[str] = Field(
        None,
        description="Unique name for state. Only needed if start_value is set (!= None). Initialization FMU parameters will be generated using this name",
        examples=["initialWindSpeed", "initialWindDirection"],
    )
    description: Optional[str] = Field(None, description="Short FMU variable description.")
    start_value: Optional[float] = Field(
        None,
        description="The default value of the parameter used for initialization. If this field is set parameters for initialization will be automatically generated for these states.",
    )
    initialization_variable: Optional[str] = Field(
        None,
        description="The name of a an input or parameter in the same model interface that should be used to initialize this state.",
    )
    agent_output_indexes: List[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^(\d+|\d+:\d+)$"),
        ]
    ] = Field(
        [],
        description="Index or range of indices of agent outputs that will be stored as internal states and will be fed as inputs in the next time step. Note: the FMU signal and the agent outputs need to have the same length.",
        examples=["10", "10:20", "30"],
    )

    @root_validator(allow_reuse=True, skip_on_failure=True)
    def check_only_one_initialization(cls, values: Dict[str, Any]):
        init_var = "initialization_variable" in values and values["initialization_variable"] is not None
        name = "name" in values and values["name"] is not None
        start_value = "start_value" in values and values["start_value"] is not None

        if init_var and (start_value or name):
            raise ValueError(
                "Only one state initialization method is allowed to be used at a time: initialization_variable cannot be set if either start_value or name is set."
            )
        if (not start_value) and name:
            raise ValueError(
                "name is set without start_value being set. Both fields needs to be set for the state initialization to be valid"
            )
        if start_value and (not name):
            raise ValueError(
                "start_value is set without name being set. Both fields needs to be set for the state initialization to be valid"
            )
        return values


class InputVariable(Variable):
    agent_input_indexes: List[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^(\d+|\d+:\d+)$"),
        ]
    ] = Field(
        [],
        description="Index or range of indices of agent inputs to which this FMU signal shall be linked to. Note: the FMU signal and the agent inputs need to have the same length.",
        examples=["10", "10:20", "30"],
    )


class OutputVariable(Variable):
    agent_output_indexes: List[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^(\d+|\d+:\d+)$"),
        ]
    ] = Field(
        [],
        description="Index or range of indices of agent outputs that will be linked to this output signal. Note: the FMU signal and the agent outputs need to have the same length.",
        examples=["10", "10:20", "30"],
    )


@dataclass
class FmiInputVariable(InputVariable):
    causality: FmiCausality
    variable_references: List[int] = []
    agent_state_init_indexes: List[int] = []

    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self.causality = kwargs.get("causality", FmiCausality.INPUT)  # type: ignore
        self.variable_references = kwargs.get("variable_references", [])  # type: ignore


@dataclass
class FmiOutputVariable(OutputVariable):
    causality: FmiCausality
    variable_references: List[int] = []

    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self.causality = kwargs.get("causality", FmiCausality.OUTPUT)  # type: ignore
        self.variable_references = kwargs.get("variable_references", [])  # type: ignore


@dataclass
class FmiVariable:
    name: str = ""
    variable_reference: int = 0
    type: FmiVariableType = FmiVariableType.REAL
    start_value: Union[bool, str, int, float] = 0
    causality: FmiCausality = FmiCausality.INPUT
    description: str = ""
    variability: FmiVariability = FmiVariability.CONTINUOUS


class ModelComponent(BaseModelConfig):
    name: str = Field(None, description="The name of the simulation model.")
    version: str = Field("0.0.1", description="The version number of the model.")
    author: Optional[str] = Field(None, description="Name or email of the model's author.")
    description: Optional[str] = Field("", description="Brief description of the model.")
    copyright: Optional[str] = Field(None, description="Copyright line for use in full license text.")
    license: Optional[str] = Field(None, description="License text or file name (relative to source files)")
    inputs: List[InputVariable] = Field(
        [],
        description="List of input signals of the simulation model.",
        examples=[[create_fmu_signal_example()]],
    )
    outputs: List[OutputVariable] = Field(
        [],
        description="List of output signals of the simulation model.",
        examples=[[create_fmu_signal_example()]],
    )
    parameters: List[InputVariable] = Field(
        [],
        description="List of parameter signals of the simulation model.",
        examples=[[create_fmu_signal_example()]],
    )
    states: List[InternalState] = Field(
        [],
        description="Internal states that will be stored in the simulation model's memory, these will be passed as inputs to the agent in the next time step.",
    )
    uses_time: Optional[bool] = Field(
        False,
        description="Whether the agent consumes time data from co-simulation algorithm.",
    )


class FmiModel:
    name: str = ""
    guid: Optional[UUID] = None
    inputs: List[FmiInputVariable] = []
    outputs: List[FmiOutputVariable] = []
    parameters: List[FmiInputVariable] = []
    author: Optional[str] = None
    version: str = "0.0.1"
    description: Optional[str] = None
    copyright: Optional[str] = None
    license: Optional[str] = None

    def __init__(self, model: ModelComponent):
        # Assign model specification to a valid FMU component complaint with FMISlave
        self.name = model.name
        self.author = model.author
        self.version = model.version
        self.description = model.description
        self.copyright = model.copyright
        self.license = model.license

        self.add_variable_references(model.inputs, model.parameters, model.outputs)
        self.add_state_initialization_parameters(model.states)

    def add_variable_references(
        self,
        inputs: List[InputVariable],
        parameters: List[InputVariable],
        outputs: List[OutputVariable],
    ):
        """Assign variable references to inputs, parameters and outputs from user interface to FMU model class.

        Args:
            inputs (List[InputVariable]): List of input variables from JSON interface
            parameters (List[InputVariable]): List of model parameters from JSON interface
            outputs (List[InputVariable]): List of output variables from JSON interface

        Returns
        -------
            A dictionary that maps variable references to FmiVariables (same as Variable but contains causality)
        """
        current_var_ref = 0
        fmu_inputs: List[FmiInputVariable] = []
        fmu_parameters: List[FmiInputVariable] = []
        fmu_outputs: List[FmiOutputVariable] = []

        for var in inputs:
            var_port_refs = []

            if var.is_array:
                # If array then allocate space for every element
                vector_port_length = var.length or 1
                var_port_refs = list(range(current_var_ref, current_var_ref + vector_port_length))
            else:
                var_port_refs = [current_var_ref]

            # Set current variable reference based on number of ports used by this input (array or scalar port)
            current_var_ref = current_var_ref + len(var_port_refs)
            fmi_variable = FmiInputVariable(
                causality=FmiCausality.INPUT,
                variable_references=var_port_refs,
                **var.__dict__,
            )
            fmu_inputs.append(fmi_variable)

        for var in parameters:
            var_port_refs = []

            if var.is_array:
                # If array then allocate space for every element
                vector_port_length = var.length or 1
                var_port_refs = list(range(current_var_ref, current_var_ref + vector_port_length))
            else:
                var_port_refs = [current_var_ref]

            # Set current variable reference based on number of ports used by this input (array or scalar port)
            current_var_ref = current_var_ref + len(var_port_refs)
            fmi_variable = FmiInputVariable(
                causality=FmiCausality.PARAMETER,
                variable_references=var_port_refs,
                **var.__dict__,
            )
            fmu_parameters.append(fmi_variable)

        for var in outputs:
            var_port_refs = []

            if var.is_array:
                # If array then allocate space for every element
                vector_port_length = var.length or 1
                var_port_refs = list(range(current_var_ref, current_var_ref + vector_port_length))
            else:
                var_port_refs = [current_var_ref]

            # Set current variable reference based on number of ports used by this input (array or scalar port)
            current_var_ref = current_var_ref + len(var_port_refs)
            fmi_variable = FmiOutputVariable(
                causality=FmiCausality.OUTPUT,
                variable_references=var_port_refs,
                **var.__dict__,
            )
            fmu_outputs.append(fmi_variable)

        self.inputs = fmu_inputs
        self.outputs = fmu_outputs
        self.parameters = fmu_parameters

    def add_state_initialization_parameters(self, states: List[InternalState]):
        """Generate or modifies FmuInputVariables for initialization of states for the InternalState objects that have set start_value and name or have set initialization_variable. Any generated parameters are appended to self.parameters.

        Args:
            states (List[InternalState]): List of states from JSON interface

        """
        init_parameters: List[FmiInputVariable] = []

        value_reference_start = (
            self.get_total_variable_number()
        )  # TODO: Biggest used value reference + 1, will this always be correct?
        current_state_index_state = 0
        for i, state in enumerate(states):
            length = len(range_list_expanded(state.agent_output_indexes))
            if state.initialization_variable is not None:
                variable_name = state.initialization_variable
                variable_name_input_index = [i for i, inp in enumerate(self.inputs) if inp.name == variable_name]
                variable_name_parameter_index = [
                    i for i, param in enumerate(self.parameters) if param.name == variable_name
                ]
                if len(variable_name_input_index) + len(variable_name_parameter_index) > 1:
                    raise ValueError(
                        f"Found {len(variable_name_input_index) + len(variable_name_parameter_index)} FMU inputs or parameters with same name (={variable_name}) when trying to use for state initialization. Variables must have a unique name."
                    )

                if len(variable_name_input_index) + len(variable_name_parameter_index) == 0:
                    raise ValueError(
                        f"Did not find any FMU variables for use for initialization with name={variable_name} for state with agent_output_indexes={state.agent_output_indexes}."
                    )
                agent_state_init_indexes = list(range(current_state_index_state, current_state_index_state + length))

                if len(variable_name_input_index) == 1:
                    self.inputs[variable_name_input_index[0]].agent_state_init_indexes = agent_state_init_indexes
                if len(variable_name_parameter_index) == 1:
                    self.parameters[
                        variable_name_parameter_index[0]
                    ].agent_state_init_indexes = agent_state_init_indexes

            elif state.start_value is not None:
                if state.name is None:
                    raise ValueError(
                        f"State with index {i} has state_value (!= None) without having a name. Either give it a name or set start_value = None"
                    )
                value_references = list(range(value_reference_start, value_reference_start + length))
                is_array = length > 1
                init_param = FmiInputVariable(
                    name=state.name,
                    description=state.description,
                    start_value=state.start_value,
                    variability=FmiVariability.FIXED,
                    type=FmiVariableType.REAL,
                    causality=FmiCausality.PARAMETER,
                    variable_references=value_references,
                    length=length,
                    is_array=is_array,
                    agent_input_indexes=[],
                    agent_state_init_indexes=list(range(current_state_index_state, current_state_index_state + length)),
                )
                init_parameters.append(init_param)
                value_reference_start += length
            current_state_index_state += length
        self.parameters = [*self.parameters, *init_parameters]

    def format_fmi_variable(self, var: Union[FmiInputVariable, FmiOutputVariable]) -> List[FmiVariable]:
        """Get an inclusive list of variables from an interface variable definition.
           Vectors are separated as N number of signals, being N the size of the array.

        Args:
            var (FmiInputVariable, FmiOutputVariable): Interface variable definition with the variable references.

        Returns
        -------
            A list of FMI formatted variables.
        """
        variables: List[FmiVariable] = []

        if var.is_array:
            for idx, var_ref in enumerate(var.variable_references):
                # Create port names that contain the index starting from 1. E.i signal[1], signal[2] ...
                name = f"{var.name}[{idx+1}]"
                fmi_var = FmiVariable(
                    name=name,
                    variable_reference=var_ref,
                    causality=var.causality,
                    description=var.description or "",
                    variability=var.variability
                    or (
                        FmiVariability.CONTINUOUS if var.causality != FmiCausality.PARAMETER else FmiVariability.TUNABLE
                    ),
                )
                variables.append(fmi_var)
        else:
            # Create a single variable in case it's not a vector port
            fmi_var = FmiVariable(
                name=var.name,
                variable_reference=var.variable_references[0],
                causality=var.causality,
                description=var.description or "",
                variability=var.variability
                or (FmiVariability.CONTINUOUS if var.causality != FmiCausality.PARAMETER else FmiVariability.TUNABLE),
                start_value=var.start_value or 0,
                type=var.type or FmiVariableType.REAL,
            )
            variables.append(fmi_var)

        return variables

    def get_fmi_model_variables(self) -> List[FmiVariable]:
        """Get a full list of all variables in ths FMU, including each index of vector ports."""
        variables = [*self.inputs, *self.parameters, *self.outputs]
        fmi_variables = list(map(lambda var: self.format_fmi_variable(var), variables))

        flat_vars = [var_j for var_i in fmi_variables for var_j in var_i]
        return flat_vars

    def get_template_mapping(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Calculate the index to value reference mapping between onnx inputs/outputs/state to fmu variables.

        Returns
        -------
            Tuple of lists of mappings between onnx indexes to fmu variables. (input_mapping, output_mapping, state_init_mapping)
        """
        # Input and output mapping in the form of agent index and fmu variable reference pairs
        input_mapping: List[Tuple[int, int]] = []
        output_mapping: List[Tuple[int, int]] = []
        state_init_mapping: List[Tuple[int, int]] = []

        for inp in self.inputs + self.parameters:
            input_indexes = range_list_expanded(inp.agent_input_indexes)
            for variable_index, input_index in enumerate(input_indexes):
                input_mapping.append((input_index, inp.variable_references[variable_index]))

            for variable_index, state_init_index in enumerate(inp.agent_state_init_indexes):
                if variable_index >= len(inp.variable_references):
                    break
                state_init_mapping.append((state_init_index, inp.variable_references[variable_index]))

        for out in self.outputs:
            output_indexes = range_list_expanded(out.agent_output_indexes)
            for variable_index, output_index in enumerate(output_indexes):
                output_mapping.append((output_index, out.variable_references[variable_index]))

        input_mapping = sorted(input_mapping, key=lambda inp: inp[0])
        output_mapping = sorted(output_mapping, key=lambda out: out[0])
        return input_mapping, output_mapping, state_init_mapping

    def get_total_variable_number(self) -> int:
        """Calculate the total amount of variables including every index of vector ports."""
        all_fmi_variables: List[Union[FmiInputVariable, FmiOutputVariable]] = [
            *self.inputs,
            *self.parameters,
            *self.outputs,
        ]
        num_variables = reduce(
            lambda prev, current: prev + len(current.variable_references),
            all_fmi_variables,
            0,
        )
        return num_variables
