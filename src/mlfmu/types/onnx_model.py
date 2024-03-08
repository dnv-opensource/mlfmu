import os
from pathlib import Path
from typing import Any, List, Union

import onnxruntime
from onnxruntime import InferenceSession

"""
	ONNX Metadata class

	Allows to import the ONNX file and figure out the input/output sizes.
"""


class ONNXModel:
    filename: str = ""
    states_name: str = ""
    state_size: int = 0
    input_name: str = ""
    input_size: int = 0
    output_name: str = ""
    output_size: int = 0
    time_input_name: str = ""
    time_input: bool = False
    __onnx_path: Path
    __onnx_session: InferenceSession

    def __init__(self, onnx_path: Union[str, os.PathLike[str]], time_input: bool = False):
        # Load ONNX file into memory
        self.__onnx_path = onnx_path if isinstance(onnx_path, Path) else Path(onnx_path)
        self.__onnx_session = onnxruntime.InferenceSession(onnx_path)

        # Assign model parameters
        self.filename = f"{self.__onnx_path.stem}.onnx"
        self.time_input = time_input

        self.load_inputs()
        self.load_outputs()

    def load_inputs(self):
        # Get inputs from ONNX file
        inputs: List[Any] = self.__onnx_session.get_inputs()
        input_names = [inp.name for inp in inputs]  # type: ignore No typing support provided by ONNX library
        input_shapes = [inp.shape for inp in inputs]  # type: ignore
        self.input_name = input_names[0]
        self.input_size = input_shapes[0][1]

        # Number of internal states
        num_states = 0

        # Based on number of inputs work out which are INTERNAL STATES, INPUTS and TIME DATA
        if len(input_names) == 3:
            self.states_name = input_names[1]
            self.time_input_name = input_names[2]
            num_states = input_shapes[1][1]
            if not self.time_input:
                raise ValueError(
                    "The ml model has 3 inputs, but the usesTime flag is set to false in the json interface. A model can only have 3 inputs if it uses time input."
                )
        elif len(input_names) == 2:
            if self.time_input:
                self.time_input_name = input_names[1]
            else:
                self.states_name = input_names[1]
                num_states = input_shapes[1][1]

        elif len(input_names) == 0 or len(input_names) > 3:
            raise ValueError(f"The number of inputs to the ml model (={len(input_names)}) must be 1, 2 or 3")

        self.state_size = num_states

    def load_outputs(self):
        # Get outputs from ONNX file
        outputs: List[Any] = self.__onnx_session.get_outputs()
        output_names = [out.name for out in outputs]  # type: ignore

        if len(output_names) != 1:
            raise ValueError(f"The number of outputs from the ml model (={len(output_names)}) must be exactly 1")

        output_shapes = [out.shape for out in outputs]  # type: ignore
        self.output_name = output_names[0]
        self.output_size = output_shapes[0][1]
