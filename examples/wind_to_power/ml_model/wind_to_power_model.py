import tensorflow as tf


class PowerPredictor(tf.keras.Model):
    """
    A keras model for predicting the power output of a wind turbine given wind speed and direction.

    Args:
        num_inputs (int): Number of input features. Default is 2.
        num_outputs (int): Number of output features. Default is 1.
        input_shift (tf.Tensor): Input shift tensor. Default is None.
        input_scale (tf.Tensor): Input scale tensor. Default is None.
        output_shift (tf.Tensor): Output shift tensor. Default is None.
        output_scale (tf.Tensor): Output scale tensor. Default is None.
    """

    def __init__(
        self,
        num_inputs: int = 2,
        num_outputs: int = 1,
        input_shift: tf.Tensor | None = None,
        input_scale: tf.Tensor | None = None,
        output_shift: tf.Tensor | None = None,
        output_scale: tf.Tensor | None = None,
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

        # 2 hidden layers, 32 units each
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        # output layer
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

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
        Set the shift and scale variables for input and output.

        Args:
            input_shift (tf.Tensor): Input shift tensor.
            input_scale (tf.Tensor): Input scale tensor.
            output_shift (tf.Tensor): Output shift tensor.
            output_scale (tf.Tensor): Output scale tensor.
            reset (bool): Whether to reset the variables to not doing any shifting or scaling. Default is False.
        """
        if reset or (self.input_shift is None and input_shift is None):
            input_shift = tf.zeros((1, self.num_inputs))
        if reset or (self.input_scale is None and input_scale is None):
            input_scale = tf.ones((1, self.num_inputs))

        if reset or (self.output_shift is None and output_shift is None):
            output_shift = tf.zeros((1, self.num_outputs))
        if reset or (self.output_scale is None and output_scale is None):
            output_scale = tf.ones((1, self.num_outputs))

        if input_shift is not None:
            self.input_shift = tf.Variable(tf.cast(input_shift, tf.float32), trainable=False, name="input_shift")
        if input_scale is not None:
            self.input_scale = tf.Variable(tf.cast(input_scale, tf.float32), trainable=False, name="input_scale")

        if output_shift is not None:
            self.output_shift = tf.Variable(tf.cast(output_shift, tf.float32), trainable=False, name="output_shift")
        if output_scale is not None:
            self.output_scale = tf.Variable(tf.cast(output_scale, tf.float32), trainable=False, name="output_scale")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform a forward pass through the model.

        Predicts the power output given wind speed and direction.

        Args:
            inputs (tf.Tensor): Input tensor (wind data)

        Returns:
            tf.Tensor: Output tensor (predicted power output)
        """
        ## model forward pass
        # normalize input
        inputs_transformed = (inputs - self.input_shift) * self.input_scale

        # Pass through dense layers
        x = self.dense1(inputs_transformed)
        x = self.dense2(x)
        output = self.dense3(x)

        # Undo normalization of output
        output_transformed = (output - self.output_shift) * self.output_scale
        return output_transformed


def create_dataset(
    wind_data: tf.Tensor, power_data: tf.Tensor, batch_size: int = 64, shuffle_buffer: int | None = None
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create a dataset from wind and power data.

    Args:
        wind_data (tf.Tensor): Wind data tensor.
        power_data (tf.Tensor): Power data tensor.
        batch_size (int): Batch size for the dataset. Default is 64.
        shuffle_buffer (int): Shuffle buffer size.

    Returns:
        tuple: A tuple containing training and validation datasets.
    """
    if shuffle_buffer is None:
        shuffle_buffer = power_data.shape[0]

    wind_dataset = tf.data.Dataset.from_tensor_slices(wind_data)
    power_dataset = tf.data.Dataset.from_tensor_slices(power_data)

    dataset = tf.data.Dataset.zip((wind_dataset, power_dataset))
    # split datasets in training set (80%) and validation set
    train, val = tf.keras.utils.split_dataset(dataset, 0.8, shuffle=True)
    return train.shuffle(shuffle_buffer).batch(batch_size), val.shuffle(shuffle_buffer).batch(batch_size)
