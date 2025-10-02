# %%
# Imports
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wind_to_power_model

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
# Training Power Predictor
train_power, val_power = wind_to_power_model.create_dataset(wind_norm, power_norm)

# %%
# Create wind_to_power model
power_model = wind_to_power_model.PowerPredictor()
power_model.set_shift_and_scale_variables(reset=True)
power_model(tf.zeros((1, power_model.num_inputs)))
power_model.summary()

power_model_name = "power"
power_path = Path(MODEL_FOLDER, f"{power_model_name}")
power_path_h5 = Path(MODEL_FOLDER, f"{power_model_name}.h5")
if os.path.isfile(power_path_h5):  # noqa: PTH113
    power_model.load_weights(power_path_h5)

# %%
# Train wind_to_power model

# compile: configure model for training
# add loss function and optimizer
# (Keras simplest; optimizer="adam" (stochastic gradient descent),
#                  loss="mae" (mean absolute error)
power_model.compile(optimizer=tf.optimizers.RMSprop(), loss="mse", metrics=["mae"])
# remove scaling and shifting of data
power_model.set_shift_and_scale_variables(reset=True)
# train the model
power_model.fit(train_power, validation_data=val_power, epochs=num_epochs_power)

# Check training progress (verify under-/overfitting)
if num_epochs_power > 0:
    loss_history = pd.DataFrame(power_model.history.history)
    loss_history.plot(title="Loss history wind_to_power model training.")

# %%
# Post training model modifications

# Account for normalization during training
power_model.set_shift_and_scale_variables(wind_mean, 1 / wind_std, -power_mean / power_std, power_std)
power_model(tf.zeros((1, power_model.num_inputs)))
power_model.save_weights(power_path_h5)
power_model.save(power_path)


# %%
# Predict to check performance
power_prediction = power_model.predict(wind_data)

print("\nDone training. Close all figures to end this program")

plt.figure()
plt.title("Power data (-) and Power Prediction (--)")
plt.plot(power_data)
plt.plot(power_prediction, linestyle="--")

error = power_data - power_prediction
plt.figure()
plt.title("Error between Power data and Power Prediction")
plt.plot(error)

plt.figure()
plt.title("Histogram of log10 of abs. error between Power data & Prediction")
plt.hist(np.log10(np.abs(error)))

plt.show()
