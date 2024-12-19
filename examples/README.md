# Examples

This folder contains a couple of example files to use for testing the MLFMU tools and some FMUs that have been generated using this tool.
Each folder is set up to contain:

* `config`: containing the `.onnx` and `interface.json` files
* `generated_fmu`: binary FMU files, to serve as examples
* `<FmuName>`: source files for the FMU

The FMUs in this folder have been validated using [FMU_check].

For further documentation of the `mlfmu` tool, see [README.md](../README.md) or the [docs] on GitHub pages.

<!-- Markdown link & img dfn's -->
[FMU_check]: https://fmu-check.herokuapp.com/
[docs]: https://dnv-opensource.github.io/mlfmu/
