from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from pydantic.fields import Field

from mlfmu.utils.signals import range_list_expanded
from mlfmu.utils.strings import to_camel

if TYPE_CHECKING:
    from uuid import UUID


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

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class Variable(BaseModelConfig):
    """Pydantic model representing a variable in an FMU component."""

    #: Unique name for the port.
    name: str = Field(
        description="Unique name for the port.",
        examples=["windSpeed", "windDirection"],
    )
    #: Data type as defined by FMI standard.
    #: Defaults to Real.
    type: FmiVariableType = Field(
        default=FmiVariableType.REAL,
        description="Data type as defined by FMI standard, defaults to Real.",
        examples=[FmiVariableType.REAL, FmiVariableType.INTEGER],
    )
    #: Short FMU variable description.
    #: Optional.
    description: str | None = Field(
        default=None,
        description="Short FMU variable description.",
    )
    #: Signal variability as defined by FMI.
    #: Optional.
    variability: FmiVariability | None = Field(
        default=None,
        description="Signal variability as defined by FMI.",
    )
    #: Initial value of the signal at time step 1.
    #: Type should match the variable type.
    #: Defaults to 0.
    start_value: float | str | bool | int | None = Field(
        default=0,
        description="Initial value of the signal at time step 1. Type should match the variable type.",
    )
    #: When dealing with an array signal, it is essential to specify the LENGTH parameter.
    #: Arrays are indexed starting from 0, and FMU signals will be structured as
    #: SIGNAL_NAME[0], SIGNAL_NAME[1], and so forth.
    #: Defaults to False.
    is_array: bool = Field(
        default=False,
        description=(
            "When dealing with an array signal, it is essential to specify the LENGTH parameter. "
            "Arrays are indexed starting from 0, and FMU signals will be structured as "
            "SIGNAL_NAME[0], SIGNAL_NAME[1], and so forth. "
            "By default, this feature is set to False."
        ),
    )
    #: Defines the number of entries in the signal if the signal is array.
    #: Optional.
    length: int | None = Field(
        default=None,
        description="Defines the number of entries in the signal if the signal is array.",
        examples=[3, 5],
    )


class InternalState(BaseModelConfig):
    """Pydantic model representing an internal state of an FMU component."""

    #: Unique name for the state. Only needed if start_value is set (!= None).
    #: Initialization FMU parameters will be generated using this name.
    #: Optional.
    name: str | None = Field(
        default=None,
        description=(
            "Unique name for state. Only needed if start_value is set (!= None). "
            "Initialization FMU parameters will be generated using this name"
        ),
        examples=["initialWindSpeed", "initialWindDirection"],
    )
    #: Short description of the FMU variable.
    #: Optional.
    description: str | None = Field(
        default=None,
        description="Short FMU variable description.",
    )
    #: The default value of the parameter used for initialization.
    #: If this field is set, parameters for initialization will be automatically generated for these states.
    #: Optional.
    start_value: float | None = Field(
        default=None,
        description=(
            "The default value of the parameter used for initialization. "
            "If this field is set parameters for initialization will be automatically "
            "generated for these states."
        ),
    )
    #: The name of an input or parameter in the same model interface that should be used to initialize this state.
    #: Optional.
    initialization_variable: str | None = Field(
        default=None,
        description=(
            "The name of a an input or parameter in the same model interface "
            "that should be used to initialize this state."
        ),
    )
    #: Index or range of indices of agent (ONNX model) outputs that will be stored as internal states
    #: and will be fed as inputs in the next time step.
    #: Note: the FMU signal and the ONNX (agent) outputs need to have the same length.
    #: Defaults to an empty list.
    agent_output_indexes: list[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^(\d+|\d+:\d+)$"),
        ]
    ] = Field(
        default=[],
        description=(
            "Index or range of indices of agent outputs that will be stored as internal states "
            "and will be fed as inputs in the next time step. "
            "Note: the FMU signal and the agent outputs need to have the same length."
        ),
        examples=["10", "10:20", "30"],
    )

    # TODO @KristofferSkare: Change return type to `Self` (from `typing`module)
    #      once we drop support for Python 3.10
    #      (see https://docs.python.org/3/library/typing.html#typing.Self)
    #      CLAROS, 2024-10-15
    @model_validator(mode="after")
    def check_only_one_initialization(self) -> InternalState:
        """
        Check if only one state initialization method is used at a time.

        Raises a ValueError if multiple state initialization methods are used simultaneously.

        Returns
        -------
            self: The FMU component instance.

        Raises
        ------
            ValueError: If initialization_variable is set and either start_value or name is also set.
            ValueError: If name is set without start_value being set.
            ValueError: If start_value is set without name being set.
        """
        init_var = self.initialization_variable is not None
        name = self.name is not None
        start_value = self.start_value is not None

        if init_var and (start_value or name):
            raise ValueError(
                "Only one state initialization method is allowed to be used at a time: "
                "initialization_variable cannot be set if either start_value or name is set."
            )
        if (not start_value) and name:
            raise ValueError(
                "name is set without start_value being set. "
                "Both fields need to be set for the state initialization to be valid."
            )
        if start_value and (not name):
            raise ValueError(
                "start_value is set without name being set. "
                "Both fields need to be set for the state initialization to be valid."
            )
        return self


class InputVariable(Variable):
    """
    Represents an input variable for an FMU component.

    Examples
    --------
        An example of `agent_input_indexes` can be ["10", "10:20", "30"].
    """

    #: Index or range of indices of ONNX (agent) inputs to which this FMU signal shall be linked to.
    #: Note: The FMU signal and the ONNX (agent) inputs need to have the same length.
    #: Defaults to an empty list.
    agent_input_indexes: list[
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                to_upper=True,
                pattern=r"^(\d+|\d+:\d+)$",
            ),
        ]
    ] = Field(
        default=[],
        description=(
            "Index or range of indices of agent inputs to which this FMU signal shall be linked to. "
            "Note: the FMU signal and the agent inputs need to have the same length."
        ),
        examples=["10", "10:20", "30"],
    )


class OutputVariable(Variable):
    """
    Represents an output variable in the FMU component.

    Examples
    --------
        An example of `agent_output_indexes` can be ["10", "10:20", "30"].
    """

    #: Index or range of indices of agent outputs that will be linked to this output signal.
    #: Note: The FMU signal and the agent outputs need to have the same length.
    #: Defaults to an empty list.
    agent_output_indexes: list[
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                to_upper=True,
                pattern=r"^(\d+|\d+:\d+)$",
            ),
        ]
    ] = Field(
        default=[],
        description=(
            "Index or range of indices of agent outputs that will be linked to this output signal. "
            "Note: The FMU signal and the agent outputs need to have the same length."
        ),
        examples=["10", "10:20", "30"],
    )


# TODO @KristofferSkare: The `FmiInputVariable` and `FmiOutputVariable` classes are marked as `@dataclass`,
#      but they inherit from (at its very root) pydantic's `BaseModel`.
#      Let's discuss whether we can possibly remove the `@dataclass` decorator.
#      Then also the manual `__init__` method in `FmiInputVariable` and `FmiOutputVariable` could be removed.
#      CLAROS, 2024-10-15
@dataclass
class FmiInputVariable(InputVariable):
    """Data class representing an input variable in an FMI component."""

    causality: FmiCausality  #: The causality of the input variable.
    variable_references: list[int]  #: The list of variable references associated with the input variable.
    #: List of state initialization indexes for ONNX model - concerns mapping of FMU input variables to ONNX states.
    agent_state_init_indexes: list[list[int]] = []  # noqa: RUF008

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Create an FMI input variable.

        Parameters
        ----------
        causality : FmiCausality, optional
            Causality of the input variable., by default FmiCausality.INPUT
        variable_references : list[int], optional
            List of variable references associated with the input variable., by default []
        """
        super().__init__(**kwargs)
        self.causality = kwargs.get("causality", FmiCausality.INPUT)
        self.variable_references = kwargs.get("variable_references", [])


@dataclass
class FmiOutputVariable(OutputVariable):
    """Data class representing an output variable in an FMI component."""

    causality: FmiCausality  #: The causality of the output variable.
    variable_references: list[int]  #: The list of variable references associated with the output variable.

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Create an FMI output variable.

        Parameters
        ----------
        causality : FmiCausality, optional
            Causality of the output variable., by default FmiCausality.OUTPUT
        variable_references : list[int], optional
            List of variable references associated with the output variable., by default []
        """
        super().__init__(**kwargs)
        self.causality = kwargs.get("causality", FmiCausality.OUTPUT)
        self.variable_references = kwargs.get("variable_references", [])


@dataclass
class FmiVariable:
    """Data class representing a variable in an FMU component."""

    name: str = ""  #: The name of the variable.
    variable_reference: int = 0  #: The reference ID of the variable.
    type: FmiVariableType = FmiVariableType.REAL  #: The type of the variable.
    start_value: bool | str | int | float = 0  #: The initial value of the variable.
    causality: FmiCausality = FmiCausality.INPUT  #: The causality of the variable.
    description: str = ""  #: The description of the variable.
    variability: FmiVariability = FmiVariability.CONTINUOUS  #: The variability of the variable.


def _create_fmu_signal_example() -> Variable:
    """
    Create an example FMU signal variable.

    Returns
    -------
    Variable
        An instance of the Variable class representing the FMU signal variable.
    """
    return Variable(
        name="dis_yx",
        type=FmiVariableType.REAL,
        description=None,
        start_value=None,
        length=None,
        variability=None,
    )


class ModelComponent(BaseModelConfig):
    """Pydantic model representing a simulation model component.

    Used to generate the JSON schema for the model interface.
    Defines the structure of the FMU and how the inputs and outputs of the ONNX model
    correspond to the FMU variables.
    """

    #: The name of the simulation model.
    name: str = Field(
        description="The name of the simulation model.",
    )
    #: The version number of the model.
    #: Defaults to "0.0.1".
    version: str = Field(
        default="0.0.1",
        description="The version number of the model.",
    )
    #: Name or email of the model's author.
    #: Optional.
    author: str | None = Field(
        default=None,
        description="Name or email of the model's author.",
    )
    #: Short description of the model.
    #: Defaults to an empty string.
    description: str | None = Field(
        default="",
        description="Short description of the model.",
    )
    #: Copyright line for use in full license text.
    #: Optional.
    copyright: str | None = Field(
        default=None,
        description="Copyright line for use in full license text.",
    )
    #: License text or file name (relative to source files).
    #: Optional.
    license: str | None = Field(
        default=None,
        description="License text or file name (relative to source files)",
    )
    #: List of input signals of the simulation model.
    #: Defaults to an empty list.
    inputs: list[InputVariable] = Field(
        default=[],
        description="List of input signals of the simulation model.",
        examples=[[_create_fmu_signal_example()]],
    )
    #: List of output signals of the simulation model.
    #: Defaults to an empty list.
    outputs: list[OutputVariable] = Field(
        default=[],
        description="List of output signals of the simulation model.",
        examples=[[_create_fmu_signal_example()]],
    )
    #: List of parameter signals of the simulation model.
    #: Defaults to an empty list.
    parameters: list[InputVariable] = Field(
        default=[],
        description="List of parameter signals of the simulation model.",
        examples=[[_create_fmu_signal_example()]],
    )
    #: Internal states that will be stored in the simulation model's memory,
    #: these will be passed as inputs to the agent in the next time step.
    #: Defaults to an empty list.
    states: list[InternalState] = Field(
        default=[],
        description=(
            "Internal states that will be stored in the simulation model's memory, "
            "these will be passed as inputs to the agent in the next time step."
        ),
    )
    #: Whether the agent consumes time data from co-simulation algorithm.
    #: Defaults to False.
    uses_time: bool | None = Field(
        default=False,
        description="Whether the agent consumes time data from co-simulation algorithm.",
    )
    #: Whether variables are allowed to be reused for state initialization when
    #: initialization_variable is used for state initialization.
    #: If set to true the variable referred to in initialization_variable will be repeated
    #: for the state initialization until the entire state is initialized.
    #: Defaults to False.
    state_initialization_reuse: bool = Field(
        default=False,
        description=(
            "Whether variables are allowed to be reused for state initialization "
            "when initialization_variable is used for state initialization. "
            "If set to true the variable referred to in initialization_variable will be repeated "
            "for the state initialization until the entire state is initialized."
        ),
    )


class FmiModel:
    """Represents an FMU model with its associated properties and variables."""

    def __init__(self, model: ModelComponent) -> None:
        """Initialize the FmiModel object with a ModelComponent object.

        Parameters
        ----------
        model : ModelComponent
            FMU component compliant with FMISlave
        """
        # Assign model specification to a valid FMU component compliant with FMISlave
        self.name: str = model.name  #: The name of the FMU model.
        self.guid: UUID | None = None  #: The globally unique identifier of the FMU model.
        self.inputs: list[FmiInputVariable] = []  #: The list of input variables for the FMU model.
        self.outputs: list[FmiOutputVariable] = []  #: The list of output variables for the FMU model.
        self.parameters: list[FmiInputVariable] = []  #: The list of parameter variables for the FMU model.
        self.author: str | None = model.author  #: The author of the FMU model.
        self.version: str = model.version or "0.0.1"  # The version of the FMU model.
        self.description: str | None = model.description  #: The description of the FMU model.
        self.copyright: str | None = model.copyright  #: The copyright information of the FMU model.
        self.license: str | None = model.license  #: The license information of the FMU model.
        #: Whether the FMU model reuses state initialization.
        self.state_initialization_reuse: bool = model.state_initialization_reuse

        self.add_variable_references(model.inputs, model.parameters, model.outputs)
        self.add_state_initialization_parameters(model.states)

    def add_variable_references(
        self,
        inputs: list[InputVariable],
        parameters: list[InputVariable],
        outputs: list[OutputVariable],
    ) -> None:
        """Assign variable references to inputs, parameters and outputs from user interface to the FMU model class.

        Parameters
        ----------
        inputs : list[InputVariable]
            List of input variables from JSON interface.
        parameters : list[InputVariable]
            List of model parameters from JSON interface.
        outputs : list[OutputVariable]
            List of output variables from JSON interface.
        """
        current_var_ref = 0
        fmu_inputs: list[FmiInputVariable] = []
        fmu_parameters: list[FmiInputVariable] = []
        fmu_outputs: list[FmiOutputVariable] = []
        fmi_variable: FmiInputVariable | FmiOutputVariable

        var: InputVariable | OutputVariable

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

    def add_state_initialization_parameters(
        self,
        states: list[InternalState],
    ) -> None:
        """Generate or modify FmuInputVariables for initialization of states.

        Generates or modifies FmuInputVariables for initialization of states for the InternalState objects
        that have set start_value and name or have set initialization_variable.
        Any generated parameters are appended to self.parameters.

        Parameters
        ----------
        states : list[InternalState]
            List of states from JSON interface.

        Raises
        ------
        ValueError
            If variables with same name are found. Variables must have a unique name.
        ValueError
            If no FMU variables were found for use for initialization.
        ValueError
            If a state has a state_value != None without having a name.
        """
        init_parameters: list[FmiInputVariable] = []

        # TODO @KristofferSkare: Biggest used value reference + 1, will this always be correct?
        value_reference_start = self.get_total_variable_number()
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
                        f"Found {len(variable_name_input_index) + len(variable_name_parameter_index)} "
                        f"FMU inputs or parameters with same name (={variable_name}) "
                        "when trying to use for state initialization. Variables must have a unique name."
                    )

                if len(variable_name_input_index) + len(variable_name_parameter_index) == 0:
                    raise ValueError(
                        "Did not find any FMU variables for use for initialization "
                        f"with name={variable_name} for state with agent_output_indexes={state.agent_output_indexes}."
                    )
                agent_state_init_indexes = list(range(current_state_index_state, current_state_index_state + length))

                if len(variable_name_input_index) == 1:
                    self.inputs[variable_name_input_index[0]].agent_state_init_indexes.append(agent_state_init_indexes)
                if len(variable_name_parameter_index) == 1:
                    self.parameters[variable_name_parameter_index[0]].agent_state_init_indexes.append(
                        agent_state_init_indexes
                    )

            elif state.start_value is not None:
                if state.name is None:
                    raise ValueError(
                        f"State with index {i} has state_value (!= None) without having a name. "
                        "Either give it a name or set start_value = None"
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
                    agent_state_init_indexes=[
                        list(range(current_state_index_state, current_state_index_state + length))
                    ],
                )
                init_parameters.append(init_param)
                value_reference_start += length
            current_state_index_state += length
        self.parameters = [*self.parameters, *init_parameters]

    def format_fmi_variable(
        self,
        var: FmiInputVariable | FmiOutputVariable,
    ) -> list[FmiVariable]:
        """Get an inclusive list of variables from an interface variable definition.

        Vectors are separated as N number of signals, being N the size of the array.

        Parameters
        ----------
        var : FmiInputVariable | FmiOutputVariable
            Interface variable definition with the variable references.

        Returns
        -------
        list[FmiVariable]
            A list of FMI formatted variables.
        """
        variables: list[FmiVariable] = []

        if var.is_array:
            for idx, var_ref in enumerate(var.variable_references):
                # Create port names that contain the index starting from 1. E.i signal[0], signal[1] ...
                name = f"{var.name}[{idx}]"
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
                start_value=var.start_value if var.start_value is not None else 0,
                type=var.type or FmiVariableType.REAL,
            )
            variables.append(fmi_var)

        return variables

    def get_fmi_model_variables(self) -> list[FmiVariable]:
        """Get a full list of all variables in the FMU, including each index of vector ports.

        Returns
        -------
        list[FmiVariable]
            List of all variables in the FMU.
        """
        variables: list[FmiInputVariable | FmiOutputVariable]
        variables = [*self.inputs, *self.parameters, *self.outputs]
        fmi_variables = [self.format_fmi_variable(var) for var in variables]

        flat_vars = [var_j for var_i in fmi_variables for var_j in var_i]
        return flat_vars

    def get_template_mapping(
        self,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
        """Calculate the index to value reference mapping between onnx inputs/outputs/state to fmu variables.

        Returns
        -------
        tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]
            Tuple of lists of mappings between onnx indexes to fmu variables.
            (input_mapping, output_mapping, state_init_mapping)
        """
        # Input and output mapping in the form of agent index and fmu variable reference pairs
        input_mapping: list[tuple[int, int]] = []
        output_mapping: list[tuple[int, int]] = []
        state_init_mapping: list[tuple[int, int]] = []

        for inp in self.inputs + self.parameters:
            input_indexes = range_list_expanded(inp.agent_input_indexes)
            input_mapping.extend(
                (input_index, inp.variable_references[variable_index])
                for variable_index, input_index in enumerate(input_indexes)
            )
            num_variable_references = len(inp.variable_references)
            for state_init_indexes in inp.agent_state_init_indexes:
                num_state_init_indexes = len(state_init_indexes)
                for variable_index, state_init_index in enumerate(state_init_indexes):
                    _variable_index = variable_index
                    if _variable_index >= num_variable_references:
                        if not self.state_initialization_reuse:
                            warnings.warn(
                                f"Too few variables in {inp.name} (={num_variable_references}) "
                                f"to initialize all states (={num_state_init_indexes}). "
                                "To initialize all states set `state_initialization_reuse=true` in interface json "
                                f"or provide a variable with length >={num_state_init_indexes}",
                                stacklevel=1,
                            )
                            break
                        _variable_index = _variable_index % num_variable_references
                    state_init_mapping.append((state_init_index, inp.variable_references[_variable_index]))

        for out in self.outputs:
            output_indexes = range_list_expanded(out.agent_output_indexes)
            output_mapping.extend(
                (output_index, out.variable_references[variable_index])
                for variable_index, output_index in enumerate(output_indexes)
            )
        input_mapping = sorted(input_mapping, key=lambda inp: inp[0])
        output_mapping = sorted(output_mapping, key=lambda out: out[0])
        return input_mapping, output_mapping, state_init_mapping

    def get_total_variable_number(self) -> int:
        """Calculate the total number of variables including every index of vector ports.

        Returns
        -------
        int
            The total number of variables.
        """
        all_fmi_variables: list[FmiInputVariable | FmiOutputVariable] = [
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
