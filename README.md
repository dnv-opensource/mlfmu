[![pypi](https://img.shields.io/pypi/v/mlfmu.svg?color=blue)](https://pypi.python.org/pypi/mlfmu)
[![versions](https://img.shields.io/pypi/pyversions/mlfmu.svg?color=blue)](https://pypi.python.org/pypi/mlfmu)
[![license](https://img.shields.io/pypi/l/mlfmu.svg)](https://github.com/dnv-opensource/mlfmu/blob/main/LICENSE)
[![docs](https://github.com/dnv-opensource/mlfmu/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/dnv-opensource/mlfmu/actions/workflows/pages/pages-build-deployment)


# mlfmu

MLFMU serves as a tool for developers looking to integrate machine learning models into simulation environments. It enables the creation of Functional Mock-up Units (FMUs), which are simulation models that adhere to the FMI standard (<https://fmi-standard.org/>), from trained machine learning models exported in the ONNX format (<https://onnx.ai/>). The mlfmu package streamlines the process of transforming ONNX models into FMUs, facilitating their use in a wide range of simulation platforms that support the FMI standard such as the [Open Simulation Platform](https://open-simulation-platform.github.io/) or DNV's [Simulation Trust Center](https://store.veracity.com/simulation-trust-center).

This work was first presented publicly at the [Open Simulation Platform conference 2024](https://opensimulationplatform.com/2024/09/23/osp-conference-2024-program/). You can find a pdf of the presentation in this repo under [`presentations/`](https://github.com/dnv-opensource/mlfmu/tree/main/presentations).

## Features

- Compile trained ML models into FMUs (Functional Mock-up Units).
- Easy to integrate in building pipelines.
- Declarative solution, just define what the inputs/outputs/parameters of your co-simulation model should look like and MLFMU will take care of the rest.
- Support for FMU signal vectors in FMI 2.0.
- Advanced customizations by enabling you to change the C++ code of the FMU.

## Installation

```sh
pip install mlfmu
```

You may need to run `conan profile detect` before running the tool. See the notes further down in this README w.r.t. required cppstd version for Windows/Linux/MacOS.

## Creating ML FMUs

### Create your own ML model

Before you use this mlfmu tool, you should create your machine learning (ML) model, using whatever your preferred tool is.

1. Define the architecture of your ML model and prepare the model to receive the inputs following to MLFMU's input format.

> Note 1: This example subclasses a Keras model for demonstration purposes. However, the tool is flexible and can accommodate other frameworks such as PyTorch, TensorFlow, Scikit-learn, and more.

> Note 2: We showcase a simple example here. For more detailed information on how you can prepare your model to be compatible with this tool, see [MLMODEL.md](MLMODEL.md)

```python
# Create your ML model
class MlModel(tf.keras.Model):
    def init(self, num_inputs = 2):
        # 1 hidden layer, 1 output layer
        self.hidden_layer = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    ...

    def call(self, all_inputs): # model forward pass
        # unpack inputs
        inputs, *_ = all_inputs

        # Do something with the inputs
        # Here we have 1 hidden layer
        d1 = self.hidden_layer(inputs)
        outputs = self.output_layer(d1)

        return outputs
    ...
```

2. Train your model, then save it as an ONNX file, e.g.:

```python
import onnx

ml_model = MlModel()
# compile: configure model for training
ml_model.compile(optimizer=tf.optimizers.RMSProp, loss='mse')
# fit: train your ML model for some number of epochs
ml_model.fit(training_dataset, epochs=nr_epochs)

# Save the trained model as ONNX at a specified path
onnx_model = tf2onnx.convert.from_keras(ml_model)
onnx.save(onnx_model, 'path/to/save')
```

3. (Optional) You may want to check your onnx file to make sure it produces the right output. You can do this by loading the onnx file and (using the same test input) compare the onnx model predictions to your original model predictions.
You can also check the model using Netron: <https://netron.app/> or <https://github.com/lutzroeder/netron>

### Preparing for and using MLFMU

Given that you have an ML model, you now need to:

1. Prepare the FMU interface specification (.json), to specify your FMU's inputs, parameters, and output, map these to the ML model's inputs and output (`agentInputIndexes`) and to specify whether it uses time (`usesTime`).

```json
// Interface.json
{
    "name": "MyMLFMU",
    "description": "A Machine Learning based FMU",
    "usesTime": true,
    "inputs": [
        {
            "name": "input_1",
            "description": "My input signal to the model at position 0",
            "agentInputIndexes": ["0"]
        },
        {
            "name": "input_2",
            "description": "My input signal as a vector with four elements at position 1 to 5",
            "agentInputIndexes": ["1:5"],
            "type": "real",
            "isArray": true,
            "length": 4
        }
    ],
    "parameters": [
        {
            "name": "parameter_1",
            "description": "My input signal to the model at position 1",
            "agentInputIndexes": ["1"]
        }
    ],
    "outputs": [
        {
            "name": "prediction",
            "description": "The prediction generated by ML model",
            "agentOutputIndexes": ["0"]
        }
    ]
}
```

More information about the interface.json schema can be found in the mlfmu\docs\interface\schema.html

2. Compile the FMU:

```sh
mlfmu build --interface-file interface.json --model-file model.onnx
```

or if the files are in your current working directory:

```sh
mlfmu build
```

## Extended documentation

For more explanation on the ONNX file structure and inputs/outputs for your model, please refer to mlfmu's [MLMODEL.md](MLMODEL.md).

For advanced usage options, e.g. editing the generated FMU source code, or using the tool via a Python class, please refer to mlfmu's [ADVANCED.md](ADVANCED.md).

## Development Setup

If you just want to use `mlfmu`, you do not need to do the following steps. This is for those aiming to do development in this repo.

### 1. Install uv

This project uses `uv` as package manager.
If you haven't already, install [uv](https://docs.astral.sh/uv), preferably using it's ["Standalone installer"](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) method: <br>
..on Windows:

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

..on MacOS and Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(see [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) for all / alternative installation methods.)

Once installed, you can update `uv` to its latest version, anytime, by running:

```sh
uv self update
```

### 2. (Windows) Install Visual Studio Build Tools

We use conan for building the FMU. For the conan building to work later on, you will need the Visual Studio Build tools 2022 to be installed. It is best to do this **before** installing conan (which gets installed as part of the package dependencies, see step 5). You can download and install the Build Tools for VS 2022 (for free) from <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>.

### 3. Clone the repository

Clone the mlfmu repository into your local development directory:

```sh
git clone https://github.com/dnv-opensource/mlfmu path/to/your/dev/mlfmu
git submodule update --init --recursive
```

### 4. Install dependencies

Run `uv sync` to create a virtual environment and install all project dependencies into it:

```sh
uv sync
```

Use the command line option `-p` to specifiy the Python version to resolve the dependencies against.
For instance, use `-p 3.12` to specify Python 3.12 .

```sh
uv sync -p 3.12
```

> Note: In case the specified Python version is not found on your machine, `uv sync` will automatically download and install it.

Optionally, use `-U` in addition to allow package upgrades. Especially in cases when you change to a newer Python version, adding `-U` can be useful. <br>
It allows the dependency resolver to upgrade dependencies to newer versions, which might be necessary to support the (newer) Python version you specified.

```sh
uv sync -p 3.12 -U
```

> Note: At this point, you should have conan installed. You will want to make sure it has the correct build profile. You can auto-detect and create the profile by running `conan profile detect`.
After this, you can check the profile:<br/>
.. on Windows: in `C:\Users\<USRNAM>\.conan2\profiles\.default` (replace `<USRNAM>` with your username). You want to have: `compiler=msvc`, `compiler.cppstd=17`, `compiler.version=193`.<br/>
.. on Linux: in `~/.conan2/profiles/default`, you will need at least `compiler.cppstd=gnu17` and `compiler.libcxx=libstdc++11`.
.. on MacOS: in `~/.conan2/profiles/default`, you will need at least `compiler.cppstd=gnu20` (tested with `compiler.version=14`).

### 5. (Optional) Activate the virtual environment

When using `uv`, most of the time there will be no longer a need to manually activate the virtual environment. <br>
Whenever you run a command via `uv run` inside your project folder structure, `uv` will find the `.venv` virtual environment in the working directory or any parent directory, and activate it on the fly:

```sh
uv run <command>
```

However, you still _can_ manually activate the virtual environment if needed.
While we did not face any issues using VS Code as IDE, you might e.g. use an IDE which needs the .venv manually activated in order to properly work. <br>
If this is the case, you can anytime activate the virtual environment using one of the "known" legacy commands: <br>
..on Windows:

```sh
.venv\Scripts\activate.bat
```

..on Linux:

```sh
source .venv/bin/activate
```

### 6. Install pre-commit hooks

The `.pre-commit-config.yaml` file in the project root directory contains a configuration for pre-commit hooks.
To install the pre-commit hooks defined therein in your local git repository, run:

```sh
uv run pre-commit install
```

All pre-commit hooks configured in `.pre-commit-config.yaml` will now run each time you commit changes.

### 7. Test that the installation works

To test that the installation works, run pytest in the project root folder:

```sh
uv run pytest
```

### 8. Run an example

```sh
cd .\examples\wind_generator\config\
uv run mlfmu build
```

As an alternative, you can run from the main directory:

```sh
uv run mlfmu build --interface-file .\examples\wind_generator\config\interface.json --model-file .\examples\wind_generator\config\example.onnx
```

_Note_: wherever you run the build command from, is where the FMU file will be created, unless you specify otherwise with `--fmu-path`.

For more options, see `uv run mlfmu --help` or `uv run mlfmu build --help`.

### 9. Use your new ML FMU

The created FMU can be used for running (co-)simulations. We have tested the FMUs that we have created in the [Simulation Trust Center], which uses the [Open Simulation Platform] software.

### 10. Compiling the documentation

This repository uses sphinx with .rst and .md files as well as Python docstrings, to document the code and usage. To locally build the docs:

```sh
cd docs
make html
```

You can then open index.html for access to all docs (for Windows: `start build\html\index.html`).

## Meta

All code in mlfmu is DNV intellectual property.

Copyright (c) 2024 [DNV](https://www.dnv.com) AS. All rights reserved.

Primary contributors:

Kristoffer Skare - [@LinkedIn](https://www.linkedin.com/in/kristoffer-skare-19606a1a1/) - <kristoffer.skare@dnv.com>

Jorge Luis Mendez - [@LinkedIn](https://www.linkedin.com/in/jorgelmh/) - <jorge.luis.mendez@dnv.com>

Additional contributors (testing, docs, examples, etc.):

Melih Akdağ - [@LinkedIn](https://www.linkedin.com/in/melih-akdag/) - <melih.akdag@dnv.com>

Stephanie Kemna - [@LinkedIn](https://www.linkedin.com/in/stephaniekemna/)

Hee Jong Park - [@LinkedIn](https://www.linkedin.com/in/heejongpark/) - <hee.jong.park@dnv.com>

## Contributing

1. Fork it (<https://github.com/dnv-opensource/mlfmu/fork>) (Note: this is currently disabled for this repo. For development, continue with the next step.)
2. Create an issue in your GitHub repo
3. Create your branch based on the issue number and type (`git checkout -b issue-name`)
4. Evaluate and stage the changes you want to commit (`git add -i`)
5. Commit your changes (`git commit -am 'place a descriptive commit message here'`)
6. Push to the branch (`git push origin issue-name`)
7. Create a new Pull Request in GitHub

For your contribution, please make sure you follow the [STYLEGUIDE](STYLEGUIDE.md) before creating the Pull Request.

### New releases

The recommended practice is to merge development PRs into main, and then, when one wants to release a new version, create a PR to push main into the release branch.

For a new release, please follow the next steps:

1. Create a PR to merge all desired changes from the `main` to the `release` branch.
1. After approval and merging, on GitHub, run the action "Bump version" on the `release` branch. This will create a new PR.
1. On GitHub, access the PR created by bump-my-version, check, and approve (if all looks good).
1. After approval and merging, on GitHub, run the action "Push to release" on the `release` branch. This will build everything and publish the package to PyPi, and create a new GitHub release.

Note; version numbers are in: `CHANGELOG.md`, `pyproject.toml`, and `docs/source/conf.py` (and should be automatically updated by bump-my-version upon release).

## Errors & fixes

- If you get an error similar to `..\fmu.cpp(4,10): error C1083: Cannot open include file: 'cppfmu_cs.hpp': No such file or directory`, you are missing cppfmu. This is a submodule to this repository. Make sure that you do a `git submodule update --init --recursive` in the top level folder.

## License & dependencies

This code is distributed under the BSD 3-Clause license. See [LICENSE](LICENSE) for more information.

It makes use of [cpp-fmu], which is distributed under the MPL license at <https://github.com/viproma/cppfmu>.

<!-- Markdown link & img dfn's -->
[mlfmu_docs]: https://dnv-opensource.github.io/mlfmu/README.html
[cpp-fmu]: https://github.com/viproma/cppfmu
[Open Simulation Platform]: https://opensimulationplatform.com/
[Simulation Trust Center]: https://store.veracity.com/simulation-trust-center
