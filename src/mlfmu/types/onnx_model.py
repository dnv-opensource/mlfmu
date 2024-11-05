import os
from pathlib import Path
from typing import Any

from onnxruntime import InferenceSession, NodeArg


class ONNXModel:
    """ONNX Metadata class.

    Represents an ONNX model and provides methods to load inputs and outputs.
    Allows to import the ONNX file and figure out the input/output sizes.

    Raises
    ------
    ValueError
        If the ml model has 3 inputs, but the `usesTime` flag is set to `false` in the json interface.
    ValueError
        If the number of inputs to the ml model is larger than 3.
    ValueError
        If the number of outputs from the ml model is not exactly 1.
    """

    filename: str = ""  #: The name of the ONNX file.
    states_name: str = ""  #: The name of the internal states input.
    state_size: int = 0  #: The size of the internal states input.
    input_name: str = ""  #: The name of the main input.
    input_size: int = 0  #: The size of the main input.
    output_name: str = ""  #: The name of the output.
    output_size: int = 0  #: The size of the output.
    time_input_name: str = ""  #: The name of the time input.
    time_input: bool = False  #: Flag indicating whether the model uses time input.
    __onnx_path: Path  #: The path to the ONNX file.
    __onnx_session: InferenceSession  #: The ONNX runtime inference session.

    def __init__(
        self,
        onnx_path: str | os.PathLike[str],
        *,
        time_input: bool = False,
    ) -> None:
        """
        Initialize the ONNXModel object by loading the ONNX file and assigning model parameters.

        Args:
            onnx_path (Union[str, os.PathLike[str]]): The path to the ONNX file.
            time_input (bool, optional): Flag indicating whether the model uses time input. Defaults to False.
        """
        # Load ONNX file into memory
        self.__onnx_path = onnx_path if isinstance(onnx_path, Path) else Path(onnx_path)
        self.__onnx_session = InferenceSession(onnx_path)

        # Assign model parameters
        self.filename = f"{self.__onnx_path.stem}.onnx"
        self.time_input = time_input

        self.load_inputs()
        self.load_outputs()

    def load_inputs(self) -> None:
        """Load the inputs from the ONNX file and assign the input name and size."""
        # Get inputs from ONNX file
        inputs: list[NodeArg] = self.__onnx_session.get_inputs()
        input_names = [inp.name for inp in inputs]
        input_shapes = [inp.shape for inp in inputs]
        self.input_name = input_names[0]
        self.input_size = input_shapes[0][1]

        # Number of internal states
        num_states = 0

        # Based on number of inputs work out which are INTERNAL STATES, INPUTS and TIME DATA
        if len(input_names) == 3:  # noqa: PLR2004
            self.states_name = input_names[1]
            self.time_input_name = input_names[2]
            num_states = input_shapes[1][1]
            if not self.time_input:
                raise ValueError(
                    "The ml model has 3 inputs, but the `usesTime` flag is set to `false` in the json interface. "
                    "A model can only have 3 inputs if it uses time input."
                )
        elif len(input_names) == 2:  # noqa: PLR2004
            if self.time_input:
                self.time_input_name = input_names[1]
            else:
                self.states_name = input_names[1]
                num_states = input_shapes[1][1]

        elif not input_names or len(input_names) > 3:  # noqa: PLR2004
            raise ValueError(f"The number of inputs to the ml model (={len(input_names)}) must be 1, 2 or 3")

        self.state_size = num_states

    def load_outputs(self) -> None:
        """Load the outputs from the ONNX file and assign the output name and size."""
        # Get outputs from ONNX file
        outputs: list[Any] = self.__onnx_session.get_outputs()
        output_names = [out.name for out in outputs]

        if len(output_names) != 1:
            raise ValueError(f"The number of outputs from the ml model (={len(output_names)}) must be exactly 1")

        output_shapes = [out.shape for out in outputs]
        self.output_name = output_names[0]
        self.output_size = output_shapes[0][1]
