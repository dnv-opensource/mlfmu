import numpy as np
import scipy.interpolate
from numpy.lib.stride_tricks import as_strided


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation.

    Parameters:
        data (numpy.ndarray): The input data to be normalized.

    Returns:
        numpy.ndarray: The normalized data.
        numpy.ndarray: The mean of the input data.
        numpy.ndarray: The standard deviation of the input data.
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)

    norm = (data - mean) / std
    return norm, mean, std


def noisy_interpolation(
    data: np.ndarray,
    upsampling: int = 10,
    spline_order: int = 3,
    noise_to_signal_ratio: float = 0.1,
    noise_window_length: int = 20,
) -> np.ndarray:
    """
    Upsample data using spline interpolation with added noise scaled to the standard deviation of the data.

    Parameters:
        data (ndarray): The input data to be interpolated.
        upsampling (int, optional): The factor by which to upsample the data. Default is 10.
        spline_order (int, optional): The order of the spline interpolation. Default is 3.
        noise_to_signal_ratio (float, optional): The ratio of noise to signal. Default is 0.1.
        noise_window_length (int, optional): The length of the window used to calculate the standard
            deviation of the data. Default is 20.

    Returns:
        ndarray: The interpolated data with added noise.
    """

    n, m = data.shape

    # Interpolate the data with a spline
    t = np.arange(n)
    spline = scipy.interpolate.make_interp_spline(t, data, axis=0, k=spline_order)  # type: ignore  # noqa: PGH003

    t_up = np.arange(n * upsampling) / upsampling
    interpolated_data = spline(t_up)

    # Add noise to the interpolation
    # Making windows of the data to calculate the windowed standard deviation
    strided_data = []
    for i in range(m):
        data_i = data[:, i]
        strided_data_i = as_strided(
            data_i,
            shape=(n - noise_window_length + 1, noise_window_length),
            strides=(data_i.strides[0], data_i.strides[0]),
        )
        strided_data.append(strided_data_i[:, :, np.newaxis])  # type: ignore  # noqa: PGH003

    strided_data = np.concatenate(strided_data, axis=-1)

    # Calculate the windowed standard deviation of the windows of the data
    windowed_std = np.std(strided_data, axis=1)
    windowed_std = np.pad(
        windowed_std, ((noise_window_length // 2, n - windowed_std.shape[0] - noise_window_length // 2), (0, 0))
    )

    # Interpolate the standard deviation of the windows with the same upscaled time as the data
    std_spline = scipy.interpolate.make_interp_spline(t, windowed_std, axis=0, k=1)  # type: ignore  # noqa: PGH003
    interpolated_std = std_spline(t_up)

    # Sample noise with the same shape as the interpolated data and scale it with the interpolated standard deviation
    white_noise = np.random.normal(0, 1, interpolated_std.shape)  # type: ignore  # noqa: NPY002, PGH003
    noise_std = np.sqrt(noise_to_signal_ratio) / upsampling * interpolated_std

    scaled_noise = white_noise * noise_std

    # Add the noise to the interpolated data
    return interpolated_data + scaled_noise
