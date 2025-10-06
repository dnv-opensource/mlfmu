import sys

import numpy as np
import tensorflow as tf


class WindGenerator(tf.keras.Model):
    """
    A keras model used for generating synthetic wind data.

    The model consists of a series of LSTM layers followed by a dense layer.

    The model predicts the derivative of the wind data, which is then used to update the wind data.

    Args:
        num_inputs (int): Number of input features. Default is 2.
        num_outputs (int): Number of output features. Default is 2.
        lstm_units (list): List of integers specifying the number of units in each LSTM layer. Default is [32,32]
        input_shift (tf.Tensor): Shift values for input normalization. Default is None.
        input_scale (tf.Tensor): Scale values for input normalization. Default is None.
        output_shift (tf.Tensor): Shift values for output normalization. Default is None.
        output_scale (tf.Tensor): Scale values for output normalization. Default is None.
        stateful (bool): Whether to use stateful LSTM layers. Default is False.

    Attributes:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output features.
        input_shift (tf.Tensor): Shift values for input normalization.
        input_scale (tf.Tensor): Scale values for input normalization.
        output_shift (tf.Tensor): Shift values for output normalization.
        output_scale (tf.Tensor): Scale values for output normalization.
        num_lstms (int): Number of LSTM layers.
        lstm_units (list): List of integers specifying the number of units in each LSTM layer.
        lstms (list): List of LSTM layers.
        output_layer (tf.keras.layers.TimeDistributed): Output layer.
        concat (tf.keras.layers.Concatenate): Concatenate layer.

    Methods:
        set_shift_and_scale_variables: Set the shift and scale variables for input and output normalization.
        call: Perform forward pass through the model.

    """

    def __init__(  # noqa: PLR0913
        self,
        num_inputs: int = 2,
        num_outputs: int = 2,
        lstm_units: list[int] = [32, 32],  # noqa: B006
        input_shift: tf.Tensor | None = None,
        input_scale: tf.Tensor | None = None,
        output_shift: tf.Tensor | None = None,
        output_scale: tf.Tensor | None = None,
        stateful: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_shift, self.input_scale, self.output_shift, self.output_scale = (
            tf.zeros((1, 1, self.num_inputs)),
            tf.ones((1, 1, self.num_inputs)),
            tf.zeros((1, 1, self.num_outputs)),
            tf.ones((1, 1, self.num_outputs)),
        )

        # LSTM layers
        self.num_lstms = len(lstm_units)
        self.lstm_units = lstm_units
        self.lstms = [
            tf.keras.layers.LSTM(
                lstm_units[i], return_sequences=True, return_state=True, stateful=stateful, name=f"LSTM_{i}"
            )
            for i in range(self.num_lstms)
        ]

        # output layer
        self.output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation=None))
        self.concat = tf.keras.layers.Concatenate()

        self.set_shift_and_scale_variables(input_shift, input_scale, output_shift, output_scale)

    def set_shift_and_scale_variables(
        self,
        input_shift: tf.Tensor | None = None,
        input_scale: tf.Tensor | None = None,
        output_shift: tf.Tensor | None = None,
        output_scale: tf.Tensor | None = None,
        reset: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """
        Set the shift and scale variables for input and output normalization.

        Args:
            input_shift (tf.Tensor): Shift values for input normalization.
            input_scale (tf.Tensor): Scale values for input normalization.
            output_shift (tf.Tensor): Shift values for output normalization.
            output_scale (tf.Tensor): Scale values for output normalization.
            reset (bool): Whether to reset the variables to not doing any shifting or scaling. Default is False.
        """

        if reset or (self.input_shift is None and input_shift is None):
            input_shift = tf.zeros((1, 1, self.num_inputs))
        if reset or (self.input_scale is None and input_scale is None):
            input_scale = tf.ones((1, 1, self.num_inputs))

        if reset or (self.output_shift is None and output_shift is None):
            output_shift = tf.zeros((1, 1, self.num_outputs))
        if reset or (self.output_scale is None and output_scale is None):
            output_scale = tf.ones((1, 1, self.num_outputs))

        if input_shift is not None:
            self.input_shift = tf.Variable(tf.cast(input_shift, tf.float32), trainable=False, name="input_shift")
        if input_scale is not None:
            self.input_scale = tf.Variable(tf.cast(input_scale, tf.float32), trainable=False, name="input_scale")

        if output_shift is not None:
            self.output_shift = tf.Variable(tf.cast(output_shift, tf.float32), trainable=False, name="output_shift")
        if output_scale is not None:
            self.output_scale = tf.Variable(tf.cast(output_scale, tf.float32), trainable=False, name="output_scale")

    def call(
        self,
        inputs: tf.Tensor,
        initial_state: list[list[tf.Tensor]] | None = None,
        return_state: bool = False,  # noqa: FBT001, FBT002
    ) -> tf.Tensor | tuple[tf.Tensor, list[list[tf.Tensor]]]:
        """
        Perform forward pass through the model.

        Args:
            inputs (tf.Tensor): Input data.
            initial_state (list): Initial state for LSTM layers.
            return_state (bool): Whether to return the LSTM states. Default is False.

        Returns:
            tf.Tensor: Output data.
            list[list[tf.Tensor]]: LSTM states (if return_state is True).
        """
        # Normalize the input data
        inputs_transformed = (inputs - self.input_shift) * self.input_scale
        xs, states = [], []
        x = inputs_transformed
        xs.append(x)
        # Pass through LSTM layers
        for i in range(self.num_lstms):
            # state_h: hidden state output, state_c: cell state
            x, state_h, state_c = self.lstms[i](
                x, initial_state=initial_state[i] if initial_state is not None else None
            )
            xs.append(x)
            states.append([state_h, state_c])
        xs_concat = self.concat(xs)

        # Pass the input and LSTM outputs through the output layer to predict the derivative
        output = self.output_layer(xs_concat)

        # Undo normalization of the predicted derivative
        output_transformed = (output - self.output_shift) * self.output_scale

        # Return the output and LSTM states if requested
        if return_state:
            return output_transformed, states

        return output_transformed


class WindGeneratorWrapper(tf.keras.Model):
    """
    A wrapper class for the WindGenerator model to make it compatible with the MlFmu tool.

    The model takes the input, state and time data
    and returns the output data containing the updated state data and the output of the FMU.
    This is done according to the MlFmu standard.

    Args:
        wind_generator (WindGenerator): WindGenerator model instance.
        batch_size (int): Batch size. Default is 1.
        signal_to_noise_ratio (float): Signal-to-noise ratio. Default is 2.0.

    Attributes:
        batch_size (int): Batch size.
        wind_generator (WindGenerator): WindGenerator model instance.
        direction_mod (tf.Tensor): Direction modifier.
        concat (tf.keras.layers.Concatenate): Concatenate layer.
        noise_std (tf.Variable): Standard deviation of noise.

    Methods:
        format_inputs: Format the input data.
        format_output: Format the output data.
        wind_inside_range: Adjust the wind values to be within a certain range.
        call: Perform forward pass through the model.

    """

    def __init__(self, wind_generator: WindGenerator, batch_size: int = 1, signal_to_noise_ratio: float = 2.0) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.wind_generator = wind_generator
        self.direction_mod = tf.constant(360.0)
        self.concat = tf.keras.layers.Concatenate()
        self.noise_std = tf.Variable(1 / tf.sqrt(signal_to_noise_ratio), trainable=False)

    def format_inputs(self, state: tf.Tensor) -> tuple[tf.Tensor, list]:
        """
        Unpack the MlFmu state into the relevant inputs to the wind generator model.

        Args:
            state (tf.Tensor): MlFmu State.

        Returns:
            tf.Tensor: Formatted wind data.
            list: Formatted LSTM states.

        """
        wind = state[:, : self.wind_generator.num_inputs]
        wind_time_expanded = tf.expand_dims(wind, axis=1)

        lstm_states_concat = state[:, self.wind_generator.num_inputs :]
        index_counter = 0
        lstm_states = []
        lstm_units = self.wind_generator.lstm_units
        for i in range(self.wind_generator.num_lstms):
            units = lstm_units[i]
            lstm_state_h = lstm_states_concat[:, index_counter : index_counter + units]
            lstm_state_c = lstm_states_concat[:, index_counter + units : index_counter + 2 * units]
            index_counter += 2 * units
            lstm_states.append([lstm_state_h, lstm_state_c])

        return wind_time_expanded, lstm_states

    def format_output(self, wind: tf.Tensor, lstm_state: list) -> tf.Tensor:
        """
        Package the result of the wind generator model into the MlFmu output format.

        Args:
            wind (tf.Tensor): Wind data.
            lstm_state (list): LSTM states.

        Returns:
            tf.Tensor: Formatted output data for the MlFmu.

        """
        # Concatenate all the lstm states into a single vector
        lstm_state_concat = self.concat([self.concat(s) for s in lstm_state])

        # Remove the time distributed dimension
        wind = tf.squeeze(wind, axis=1)

        output = self.concat([wind, lstm_state_concat])

        return output

    def wind_inside_range(self, wind: tf.Tensor) -> tf.Tensor:
        """
        Adjust the wind values to be within the valid range of what it represents.

        The wind speed is clipped to be non-negative and the wind direction is adjusted to be within the range [0, 360).

        Args:
            wind (tf.Tensor): Wind data.

        Returns:
            tf.Tensor: Adjusted wind data.

        """
        wind_speed = wind[:, :, :1]
        wind_speed_relu = tf.keras.activations.relu(wind_speed)

        wind_direction = wind[:, :, 1:]
        wind_direction_sign = tf.math.sign(wind_direction)
        wind_direction_mod_sign = tf.math.sign(wind_direction - self.direction_mod)
        wind_direction_mod = wind_direction - self.direction_mod / 2 * (wind_direction_sign + wind_direction_mod_sign)

        return self.concat([wind_speed_relu, wind_direction_mod])

    def call(self, args: list[tf.Tensor]) -> tf.Tensor:
        """
        Perform forward pass through the model.

        Predicts the next wind data and updated LSTM states based on the current wind data and LSTM states
        and a noise signal.

        Args:
            args: Input arguments from the MlFmu standard.

        Returns:
            tf.Tensor: Output data for the MlFmu containing the updated state and the output of the FMU.

        """
        # Unpack the input arguments from MlFmu standard
        inputs, state, time = args

        # Unpack the state into the relevant inputs to the wind generator model
        wind, lstm_state = self.format_inputs(state)

        # Get the time step as the second column of the time data
        dt = time[:, 1:]

        # Ensure the wind data is within the valid range
        wind_formatted = self.wind_inside_range(wind)

        # Predict the derivative of the wind data using the wind generator model
        derivative, new_lstm_state = self.wind_generator(wind_formatted, lstm_state, return_state=True)

        # The input to the FMU is a noise signal to introduce randomness in the signal generation
        noise = tf.expand_dims(inputs, axis=1)

        # Update the predicted derivative with the noise scaled by the signal-to-noise ratio and the wind
        # generator output scale
        dw = derivative + noise * self.noise_std * self.wind_generator.output_scale

        # Update the wind data using the predicted derivative with noise
        new_wind = wind_formatted + dt * dw

        # Ensure the updated wind data is within the valid range
        new_wind_formatted = self.wind_inside_range(new_wind)

        # Package the result of the wind generator model into the MlFmu output format
        output = self.format_output(new_wind_formatted, new_lstm_state)
        return output


def create_dataset(
    wind_data: np.ndarray,
    window_length: int = 32,
    dt: float = 600,
    shuffle_buffer: int | None = None,  # noqa: ARG001
    shift: int = 1,
    mean: np.ndarray | float | None = None,
    std: np.ndarray | float | None = None,
) -> tuple[tf.data.Dataset, tf.Tensor]:
    """
    Create a dataset for training the wind generator model.

    Args:
        wind_data (np.ndarray): Wind data.
        window_length (int): Length of the input/output windows. Default is 32.
        dt (float): Time step. Default is 600.
        shuffle_buffer (int): Buffer size for shuffling the dataset.
        shift (int): Shift between consecutive windows. Default is 1.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.

    Returns:
        tf.data.Dataset: Training dataset.
        tf.Tensor: Standard deviation of wind derivative data for undoing the normalization.

    """
    wind_data_denormal = wind_data * std + mean

    wind_data_diff = wind_data_denormal[1:] - wind_data_denormal[:-1]

    # Using the smallest difference in angles.
    # (E.g. diff between 350 and 10 is 20)
    wind_data_diff[:, 1] = np.mod((wind_data_diff[:, 1] + 180), 360) - 180

    wind_derivative_data = wind_data_diff / dt
    wind_derivative_std = tf.math.reduce_std(wind_derivative_data, axis=0)
    wind_derivative_data = wind_derivative_data / wind_derivative_std

    model_inputs = wind_data[:-1]
    input_dataset = tf.data.Dataset.from_tensor_slices(model_inputs)

    model_outputs = wind_derivative_data
    output_dataset = tf.data.Dataset.from_tensor_slices(model_outputs)

    input_dataset_windowed = input_dataset.window(window_length, shift=shift, drop_remainder=True)
    input_dataset_windowed = input_dataset_windowed.flat_map(lambda window: window.batch(window_length))

    output_dataset_windowed = output_dataset.window(window_length, shift=shift, drop_remainder=True)
    output_dataset_windowed = output_dataset_windowed.flat_map(lambda window: window.batch(window_length))

    dataset = tf.data.Dataset.zip((input_dataset_windowed, output_dataset_windowed))

    return dataset.shuffle(10000).batch(32), wind_derivative_std


if __name__ == "__main__":
    sys.exit()
