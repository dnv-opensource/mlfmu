# Changelog

All notable changes to the [mlfmu] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Dependencies
* updated to ruff==0.4.2  (from ruff==0.3.0)
* updated to pyright==1.1.360  (from pyright==1.1.352)
* updated to sourcery==1.16  (from sourcery==1.15)
* updated to pytest>=8.2  (from pytest>=7.4)
* updated to pytest-cov>=5.0  (from pytest-cov>=4.1)
* updated to Sphinx>=7.3  (from Sphinx>=7.2)
* updated to sphinx-argparse-cli>=1.15  (from sphinx-argparse-cli>=1.11)
* updated to myst-parser>=3.0  (from myst-parser>=2.0)
* updated to furo>=2024.4  (from furo>=2023.9.10)

### Changed
* Add conan dependency to pyproject.toml

### Changed
* Moved CMake + conan + c++ package files and folders with cpp code inside src folder to be included in package
* Replace pkg_resources with importlib.metadata for getting packages version to work in python 3.12
* Replace deprecated root_validator with model_validator in pydanitc class
* Remove unnecessary hard coded values in utils/builder.py 
* Add MlFmuBuilder class to generate code and compile FMU
  * Find default paths to files and directories if not given in cli
  * Run functions in utils/builder.py according to which command is being run
  * Clean up temporary/build files after the process is done
* Complete cli interface
  * Add subparsers to cli argparser to handle several different commands
  * Create MlFmuBuilder from args and run code according to command
* Change cli test to match the new cli interface/parser

### Changed

* Default agent_(input/output)_indexes is [] by default instead of None
* Updated doc by running publish-interface-docs
* Added feature to be able to initialize states using previously defined parameters or inputs
  * This is done by setting "initializationVariable" = "{variable name}", instead of using the "name" and "start_value" attributes
  * Added flag in schema to allow a variable to be reused when initializing states
  * Allowed for multiple states to use the same variable for initialization
  
### Changed
* Deleted azure files from old azure devops repo

### Changed
* The generated modelDescription.xml files now also contain <ModelStructure> with a list of the outputs of the FMU.

### Changed
* OnnxFmu cpp template class updated to be able to initialize state with FMU Variables
  * Add variables to specify which FMU variable that should be used to initialize which state
  * Add function to DoStep that initializes the state at the beginning of the first time step. 
* model_definitions_template.h
  * Add definitions for the the number of states that should be initialized and array of index/value reference pairs to describe how the states should be initialized
* Fmu Component json interface
  * Add name, description and start_value to state in the json interface
  * States changed from a single InternalState to a list of InternalState 
* Added code to generate parameters for initialization of state

### Added
* .clang-format to consistently format cpp code

### Changed
* Fixed typo from 'tunnable' to 'tunable'
* Fixed number of onnx output check to be correct (1-3 and not always raising exception)
* Fix correct fmi causality for parameters
* Fix correct default variability for parameters
* Fix expanding of array variables into one variable per index for inputs and parameters to work as outputs

### Changed
* Default build target in builder to wind_to_power example

### Added
* Wind to power model example in examples
  * Onnx file containing ml model
  * Interface json file containing information needed not contained in onnx file
* The generated fmu files resulting from running running builder on the new wind_to_power example

### Changed
* CMakeList.txt (from old repo) to be able to take an arbitrary path where the fmu files to be compiled are located
* Builder to run commands to compile generated files into an fmu
  * Can take and keep track of arbitrary paths
* Builder checks if the files needed to compile the fmu exists before trying to compile the fmu
* Replaced "TODO: Trow Error?" with raising a fitting exception 

### Added
* CMakeLists.txt and conanfile.txt from old repo to configure compiling/building FMU from generated files

### Changed
* replaced black formatter with ruff formatter

### Dependencies
* updated to ruff==0.3.0  (from ruff==0.2.1)
* updated to pyright==1.1.352  (from pyright==1.1.350)
* removed black

### Changed
* Changed publishing workflow to use OpenID Connect (Trusted Publisher Management) when publishing to PyPI
* Updated copyright statement
* VS Code settings: Turned off automatic venv activation

### Dependencies
* updated to dictIO>=0.3.3  (from dictIO>=0.3.1)

### Added
* README.md : Under `Development Setup`, added a step to install current package in "editable" mode, using the pip install -e option.
This removes the need to manually add /src to the PythonPath environment variable in order for debugging and tests to work.

### Removed
* VS Code settings: Removed the setting which added the /src folder to PythonPath. This is no longer necessary. Installing the project itself as a package in "editable" mode, using the pip install -e option, solves the issue and removes the need to manually add /src to the PythonPath environment variable.


## [0.1.6] - 2024-02-20

### Changed
* Moved all project configuration from setup.cfg to pyproject.toml
* Moved all tox configuration from setup.cfg to tox.ini.
* Moved pytest configuration from pyproject.toml to pytest.ini
* Deleted setup.cfg

### Dependencies
* updated to black[jupyter]==24.1  (from black[jupyter]==23.12)
* updated to ruff==0.2.1  (from ruff==0.1.8)
* updated to pyright==1.1.350  (from pyright==1.1.338)
* updated to sourcery==1.15  (from sourcery==1.14)
* updated to dictIO>=0.3.1  (from dictIO>=0.2.9)


## [0.1.5] - 2023-11-08

### Changed

* incorporated latest updates introduced in mvx


## [0.1.4] - 2023-09-25

### Dependencies

* Updated dependencies to latest versions


## [0.1.3] - 2023-08-24

### Changed

* GitHub workflow publish_release.yml: corrected smaller errors
* Explicitly removed .env file from remote repository
* Updated README.md to include guidance on how to create a .env file locally
* dependencies: updated packages in requirements-dev.txt to latest versions


## [0.1.2] - 2023-06-22

### Changed

* Modularized GitHub workflows
* requirements-dev.txt: Updated dependencies to latest versions
* setup.cfg: indicated supported Python versions as py310 and py311 <br>
  (from formerly py39 and py310)
* GitHub workflows: changed default Python version from 3.10 to 3.11


## [0.1.1] - 2023-05-02

### Changed

* requirements-dev.txt: Updated dependencies to latest versions


## [0.1.0] - 2023-02-21

### Changed

* pyproject.toml: Changed ruff configuration to by default allow Uppercase variable names in functions. <br>
(As this is a very common case in science calculus)
* README.md: Changed install infos for development setup to pip install requirements-dev.txt (not requirements.txt)


## [0.0.1] - 2023-02-21

* Initial release

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
    * function x in module z

### Removed

* following features have been removed:
    * function y in module z



<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-innersource/mlfmu/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.5...v0.1.6
[0.1.5]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.4...v0.1.5
[0.1.4]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.3...v0.1.4
[0.1.3]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.2...v0.1.3
[0.1.2]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.1...v0.1.2
[0.1.1]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/mlfmu/releases/tag/v0.0.1
[mlfmu]: https://github.com/dnv-innersource/mlfmu
