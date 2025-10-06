# %%
# Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wind_generator_model

# Add the directory two levels up to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
# so that we can import from utils.py
from utils import noisy_interpolation, normalize_data

# %%
# Getting data
# This assumes you run this file from the folder ML-models/windpower/
WINDPOWER_FOLDER = Path().absolute()
MODEL_FOLDER = Path(WINDPOWER_FOLDER, "trained_model")
# create folder, if needed
if not MODEL_FOLDER.exists():
    MODEL_FOLDER.mkdir()
data = pd.read_csv(Path(WINDPOWER_FOLDER, "../../data/T1.csv"))

wind_data = data[["Wind Speed (m/s)", "Wind Direction (°)"]].to_numpy()
power_data = data[["LV ActivePower (kW)"]].to_numpy()

# %%
# Configurations
dt = 600
upsampling = 10
dt_up = dt / upsampling

num_pred_steps = 4096

# How long should the models be trained
# Set them to 0 to just use the pre-trained models
# To retrain new models, you can set each to, for example, 10.
num_epochs_wind = 0
num_epochs_power = 10

# %%
# Data preprocessing
wind_speed_data = wind_data[:, :1]
wind_direction_data = wind_data[:, 1:]

# Interpolate wind data for the wind generator training on smaller time steps
wind_x_data = wind_speed_data * np.cos(wind_direction_data * np.pi / 180)
wind_y_data = wind_speed_data * np.sin(wind_direction_data * np.pi / 180)

wind_xy_data = np.concatenate([wind_x_data, wind_y_data], axis=-1)

# Interpolating in xy since the signals are valid for all values
wind_xy_interpolated = noisy_interpolation(wind_xy_data, upsampling=upsampling)

# Converting from xy to speed/direction for training
wind_speed_interpolated = tf.linalg.norm(wind_xy_interpolated, axis=-1)
wind_direction_interpolated = np.mod(
    tf.atan2(wind_xy_interpolated[:, 1], wind_xy_interpolated[:, 0]) * 180 / np.pi + 360, 360
)

wind_interpolated = np.concatenate(
    [wind_speed_interpolated[:, np.newaxis], wind_direction_interpolated[:, np.newaxis]], axis=-1
)

# Normalize Training Data
wind_norm, wind_mean, wind_std = normalize_data(wind_data)
power_norm, power_mean, power_std = normalize_data(power_data)

wind_interpolated_norm, wind_interpolated_mean, wind_interpolated_std = normalize_data(wind_interpolated)

# %%
# Load trained model
wind_dataset, derivative_scale = wind_generator_model.create_dataset(
    wind_interpolated_norm,
    num_pred_steps,
    shift=upsampling,
    dt=dt_up,
    mean=wind_interpolated_mean,
    std=wind_interpolated_std,
)

wind_model = wind_generator_model.WindGenerator()

wind_model_name = "wind_generator_interpolated"
wind_path = Path(MODEL_FOLDER, f"{wind_model_name}")
wind_path_h5 = Path(MODEL_FOLDER, f"{wind_model_name}.h5")

wind_model.set_shift_and_scale_variables(reset=True)
wind_model(tf.zeros((1, 1, wind_model.num_inputs)))
if wind_path_h5.is_file():
    wind_model.load_weights(wind_path_h5)

wind_model.summary()


# %%
# Train wind_generator model

# compile: configure model for training
wind_model.compile(
    optimizer=tf.optimizers.RMSprop(learning_rate=0.002),
    loss="mse",
)
# scale data
wind_model.set_shift_and_scale_variables(reset=True)
# train the model
wind_model.fit(wind_dataset, epochs=num_epochs_wind)

# Check training progress (verify under-/overfitting)
if num_epochs_wind > 0:
    loss_history = pd.DataFrame(wind_model.history.history)
    loss_history.plot(title="Loss history wind_generator model training.")

# %%
# Changes after training:
wind_model.set_shift_and_scale_variables(
    tf.reshape(wind_interpolated_mean, (1, 1, wind_interpolated_mean.shape[-1])),
    tf.reshape(1 / wind_interpolated_std, (1, 1, wind_interpolated_std.shape[-1])),
    tf.zeros((1, 1, derivative_scale.shape[-1])),
    tf.reshape(derivative_scale, (1, 1, derivative_scale.shape[-1])),
)
wind_model(tf.zeros((1, 1, wind_model.num_inputs)))
wind_model.save_weights(wind_path_h5)
wind_model.save(wind_path)


# %%
# Generate wind using wind generator
initial_state = [[tf.zeros((1, 32)), tf.zeros((1, 32))] for i in range(wind_model.num_lstms)]
initial_inputs = wind_model.input_shift + np.random.normal(0, 1, (1, 1, 2)) / wind_model.input_scale  # noqa: NPY002

wind_model_reloaded = wind_generator_model.WindGenerator()
wind_model_reloaded.set_shift_and_scale_variables(reset=True)
wind_model_reloaded(initial_inputs, initial_state, return_state=True)
wind_model_reloaded.load_weights(wind_path_h5)

state = initial_state
inputs = initial_inputs

num_predictions = 1000

time_series = []
derivatives = []

signal_to_noise_ratio = 2.0

# Create Wrapper for the trained model to integrate the derivatives outputted from the trained model
# This wrapper also makes it compatible to be converted to FMU
wind_generator_wrapped = wind_generator_model.WindGeneratorWrapper(wind_model_reloaded, 1, signal_to_noise_ratio)

wind_generator_state = wind_generator_wrapped.format_output(inputs, state)
inputs = wind_generator_state[:, :2]
time = tf.constant([[0.0, dt_up]])

time_series.append(inputs)
for _ in range(num_predictions):
    noise_input = tf.random.normal((1, 2))
    wind_generator_state = wind_generator_wrapped([noise_input, wind_generator_state, time])
    inputs = wind_generator_state[:, :2]
    time_series.append(inputs)
wind_generator_wrapped.summary()

wind_model_name = "wind_generator_wrapped"
wind_path = Path(MODEL_FOLDER, f"{wind_model_name}")
wind_path_h5 = Path(MODEL_FOLDER, f"{wind_model_name}.h5")
wind_generator_wrapped.save(wind_path)
wind_generator_wrapped.save_weights(wind_path_h5)

# wind generator (wrapped) output data
time_series = tf.squeeze(tf.concat(time_series, axis=0))
wind_speed_estimate = time_series[:, 0]
wind_direction_estimate = time_series[:, 1]

# %%
# Looking at performance

# Predict to check performance
wind_generated = tf.concat([wind_speed_estimate[:, tf.newaxis], wind_direction_estimate[:, tf.newaxis]], axis=1)

start_point = np.random.choice(np.arange(wind_data.shape[0] - num_predictions // upsampling))  # noqa: NPY002

print("\nDone training. Close all figures to end this program")

plt.figure()
plt.title("Wind Speed estimate and Wind data")
plt.subplot(1, 2, 1)
plt.plot(wind_speed_estimate)
plt.subplot(1, 2, 2)
plt.plot(wind_data[start_point : num_predictions // upsampling + start_point, 0])

plt.figure()
plt.title("Wind Direction estimate and Wind data")
plt.subplot(1, 2, 1)
plt.plot(wind_direction_estimate)
plt.subplot(1, 2, 2)
plt.plot(wind_data[start_point : num_predictions // upsampling + start_point, 1])

plt.show()
