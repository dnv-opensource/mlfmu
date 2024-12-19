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
