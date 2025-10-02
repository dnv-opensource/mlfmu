# Examples

This folder contains a couple of example files to use for testing the MLFMU tools and some FMUs that have been generated using this tool.
Each folder is set up to contain at least:

* `config`: the `.onnx` and `interface.json` files, which you can use to test the `mlfmu` commands with
* `generated_fmu`: binary FMU files, to serve as examples, as generated when running `mlfmu build` for the config files
* `<FmuName>`: files for the FMU as generated when running `mlfmu codegen`

The FMUs in this folder have been validated using [FMU_check].

For further documentation of the `mlfmu` tool, see [README.md](../README.md) or the [docs] on GitHub pages.

<!-- Markdown link & img dfn's -->
[FMU_check]: https://fmu-check.herokuapp.com/
[docs]: https://dnv-opensource.github.io/mlfmu/

## Creating the wind_generator and wind_to_power ML models, before converting to FMU

We have included the python code that was used to create the onnx models, which are then converted to FMUs. You do not need to do this yourself, since we included the onnx models in the `config` folder for each example. However, it can be insightful, in particular if you want to understand better how we implemented the ML model wrapper so that it works well for conversion to FMU.

If you want to train the models yourself and test everything, bottom up, here are the steps we took for training the models.
The below steps assume you work on Windows and use the specific versions of software specified in the requirements.txt file, we have not tested this for other systems (as this is a mere, simple example).

These models are using the publicly available ["Wind Turbine Scada Dataset"](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/data) from Kaggle, the data is included here for creating a complete example.

### Install: Conda with pip (Windows native example)

Note that Windows native only supports up to TensorFlow 2.10. Newer versions can only be installed with WSL2.
This instruction is for using TF 2.10 with Windows native and pip.

1. Install conda, e.g. miniconda: <https://docs.anaconda.com/miniconda/>

2. Create a conda (or other virtual) environment, with some required packages:

* From the `examples` directory:

```sh
# We need an older version of Python (for TensorFlow 2.10):
conda create -n mlfmu-examples python=3.10
```

* Install Tensorflow / GPU support:

```sh
conda activate mlfmu-examples
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# note: you need TensorFlow 2.10 and NumPy < 2 (TensorFlow 2.10 cannot handle NumPy 2.0)
python -m pip install "tensorflow==2.10.0" "numpy==1.23.5"
# test your install
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

* This should return an array of GPU devices, e.g.

```sh
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

3. Now we want to install more packages needed for the windpower demo:

> Note: if you did not specify the python version when you created your conda environment (which we recommend doing!), you will need to run `conda install pip` before running the next command.

```sh
# run `conda activate mlfmu-examples`, if you did not already do so
pip install -r requirements.txt
```

It is expected that you see an error about requirements for different protobuf versions, which you can safely ignore for this folder.

4. Because of some conflicts with package versions and their requirements, you may need to upgrade tf2onnx now:

```sh
pip install -U tf2onnx
```

### ML Model creation instructions

1. Make sure you are in your virtual environment, either ```conda activate mlfmu-examples```:

2. To generate the FMUs, firstly run the notebooks to train the ML models and save them by running ```training_models.py```. Go into the directory of the specific model you are interested in (e.g. `wind_to_power\ml_model`):

```sh
python train_model.py
```

This should create a (new) folder called `trained_model` with:

* wind_generator: wind_generator_interpolated (keras model) and wind_generator_interpolated.hs (model weights)
* wind_to_power: power (keras model) and power.h5 (model weights)

3. Now you can create onnx files for either the power predictor (power.onnx) or the wind generator (wind.onnx), from the respective folders:

```sh
python power_to_onnx.py
python generator_to_onnx.py
```

The resulting `.onnx` files will be stored in the `trained_model` folder.
These onnx files can then be used with the mlfmu tool to create FMUs to run in STC.
