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

saved_model_path = Path(MODEL_FOLDER, "power")
onnx_model_path = Path(MODEL_FOLDER, "power.onnx")

# %%
# Loading the saved keras Model
print(f"loading model from: {saved_model_path}")
keras_model = tf.keras.models.load_model(saved_model_path)

# %%
# Converting to onnx model
onnx_model, _ = tf2onnx.convert.from_keras(keras_model)

# %%
# Saving onnx model
onnx.save(onnx_model, onnx_model_path)

# %%
# Loading and testing saved model to see if it works
loaded_onnx_model = rt.InferenceSession(onnx_model_path)

inputs = loaded_onnx_model.get_inputs()
outputs = loaded_onnx_model.get_outputs()

input_names = [inp.name for inp in inputs]
input_shapes = [inp.shape for inp in inputs]

output_names = [out.name for out in outputs]
output_shapes = [out.shape for out in outputs]

# testing

# create test input data
num_inputs = 100
wind_speed = tf.random.uniform((num_inputs, 1), 0.0, 25.0)
wind_direction = tf.random.uniform((num_inputs, 1), 0.0, 360.0)
test_inputs = tf.concat([wind_speed, wind_direction], axis=-1)
onnx_test_inputs = {name: list(test_inputs) for name in input_names}

# make some predictions
# predict with keras model
keras_output = keras_model(test_inputs)

# predict with onnx model
onnx_output = loaded_onnx_model.run(output_names=output_names, input_feed=onnx_test_inputs)
onnx_output = tf.concat(onnx_output, axis=0)

# verify that predictions are the same(ish)

error = keras_output - onnx_output
mae = tf.math.reduce_mean(tf.abs(error), axis=0)
print(f"MAE between keras and onnx model outputs: {mae}")

# assert_allclose: Given two array_like objects,
# check that their shapes and all elements are equal..
# An exception is raised if the shapes mismatch or any values conflict.
np.testing.assert_allclose(keras_output, onnx_output, rtol=1e-3, atol=1e-3)
print("Sanity check: Results of keras and onnx model are the same! (rtol 1e-3)")
