# %%
# Imports
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import tf2onnx

# %% Defining Paths
# This assumes that you run this file from ML-models/windpower/
PATH_TO_CURRENT_FOLDER = Path().absolute()
MODEL_FOLDER = Path(PATH_TO_CURRENT_FOLDER, "trained_model")

saved_model_path = Path(MODEL_FOLDER, "wind_generator_wrapped")
onnx_model_path = Path(MODEL_FOLDER, "wind.onnx")

# %%
# Loading the saved keras Model
print(f"loading model from: {saved_model_path}")
keras_model = tf.keras.models.load_model(saved_model_path, compile=False)

# %%
# Converting to onnx model
# state length: 130 = 2 (wind) + 2*64 (lstm states)
input_signature = [
    tf.TensorSpec((None, 2), name="inputs"),
    tf.TensorSpec((None, 130), name="state"),
    tf.TensorSpec((None, 2), name="time"),
]
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature)

# %%
# Saving onnx model
onnx.save(onnx_model, onnx_model_path)

# %%
# Loading and testing saved model to see if it works
loaded_onnx_model = rt.InferenceSession(onnx_model_path)

all_inputs = loaded_onnx_model.get_inputs()
all_outputs = loaded_onnx_model.get_outputs()

input_names = [inp.name for inp in all_inputs]
input_shapes = [inp.shape for inp in all_inputs]

output_names = [out.name for out in all_outputs]
output_shapes = [out.shape for out in all_outputs]

## Testing

# create test input data
nr_test_inputs = 100

# split up inputs into its 3 elements
inputs, state, time = all_inputs

# create wind model to be able to define initial state / inputs
wind_generator_state = tf.zeros((1, *input_signature[1].shape[1:]))

# same dt as in training_models.py
dt = 600
upsampling = 10
dt_up = dt / upsampling
time = tf.constant([[0.0, dt_up]])

# iteratively make predictions (prev state is used as input for next prediction)
wind_generator_state_keras = wind_generator_state
wind_generator_state_onnx = wind_generator_state
time_series_keras = [wind_generator_state_keras[:, :2]]
time_series_onnx = [wind_generator_state_onnx[:, :2]]
for i in range(1, nr_test_inputs):
    # random noise input
    noise_input = tf.random.normal((1, 2))
    # if no prev prediction, initialize the state
    initial_inputs = [noise_input, wind_generator_state, time]
    # predict with keras
    wind_generator_state_keras = keras_model(initial_inputs)
    time_series_keras.append(wind_generator_state_keras[:, :2])
    # predict with onnx
    # onnx file needs the input in (x,) format, so reshape tensors
    initial_inputs = [
        tf.reshape(tf.transpose(noise_input), (2,)),
        tf.reshape(tf.transpose(wind_generator_state), (130,)),
        tf.reshape(tf.transpose(time), (2,)),
    ]
    onnx_inputs = {nm: [i] for nm, i in zip(input_names, initial_inputs, strict=False)}
    wind_generator_state_onnx = loaded_onnx_model.run(output_names=output_names, input_feed=onnx_inputs)
    # reshape onnx output to tensor (otherwise np array)
    wind_generator_state_onnx = tf.reshape(wind_generator_state_onnx, (1, 130))
    time_series_onnx.append(wind_generator_state_onnx[:, :2])

# model output data
keras_time_series = tf.concat(time_series_keras, axis=0)
onnx_time_series = tf.concat(time_series_onnx, axis=0)
# this results in tensors of shape (nr_exp,2)
# so we can separate out wind speed and wind direction:
# 0 = wind speed
keras_wind_speed_estimate = keras_time_series[:, 0]
onnx_wind_speed_estimate = onnx_time_series[:, 0]
# 1 = wind direction
keras_wind_direction_estimate = keras_time_series[:, 1]
onnx_wind_direction_estimate = onnx_time_series[:, 1]

error = keras_wind_speed_estimate - onnx_wind_speed_estimate
mae = tf.math.reduce_mean(tf.abs(error), axis=0)
print(f"MAE between keras and onnx model wind_speed_estimate outputs: {mae}")

error = keras_wind_direction_estimate - onnx_wind_direction_estimate
mae = tf.math.reduce_mean(tf.abs(error), axis=0)
print(f"MAE between keras and onnx model wind_direction_estimate outputs: {mae}")

# assert_allclose: Given two array_like objects,
# check that their shapes and all elements are equal..
# An exception is raised if the shapes mismatch or any values conflict.
np.testing.assert_allclose(keras_wind_speed_estimate, onnx_wind_speed_estimate, rtol=1e-3, atol=1e-3)
print("Sanity check: Results of keras and onnx model predictions for wind speed are the same (rtol 1e-3)!")
np.testing.assert_allclose(keras_wind_direction_estimate, onnx_wind_direction_estimate, rtol=1e-2, atol=1e-2)
print("Sanity check: Results of keras and onnx model predictions for wind direction are the same (rtol 1e-2)!")
