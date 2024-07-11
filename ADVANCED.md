# Advanced Usage

## Editing generated FMU source

The command `mlfmu build` will both generate the C++ source code for the mlfmu and compile it automatically. However, it is possible to split this into two steps where it is possible to edit the source code to change the behavior of the resulting FMU.

 ```sh
 mlfmu codegen --interface-file Interface.json --model-file model.onnx --fmu-source-path path/to/generated/source
 ```

This will result in a folder containing the source structured as below.

```sh
[FmuName]
├── resources
│   └── *.onnx
├── sources
│   ├── fmu.cpp
│   └── model_definitions.h
└── modelDescription.xml
```

Of these generated files, it is only recommended to modify `fmu.cpp`.
In this file one can e.g. modify the `DoStep` function of the generated FMU class.

```cpp
class FmuName : public OnnxFmu
{
public:
    FmuName(cppfmu::FMIString fmuResourceLocation)
        : OnnxFmu(fmuResourceLocation)
    { }

    bool DoStep(cppfmu::FMIReal currentCommunicationPoint, cppfmu::FMIReal dt, cppfmu::FMIBoolean newStep,
        cppfmu::FMIReal& endOfStep) override
    {
        // Implement custom behavior here
        // ...

        // Call the base class implementation
        return OnnxFmu::DoStep(currentCommunicationPoint, dt, newStep, endOfStep);
    }
private:
};
```

After doing the modification to the source code, one can simply run the `compile` command to complete the process.

```sh
mlfmu compile --fmu-source-path path/to/generated/source
```

## Using class

In addition to the command line interface, one can use the same functionality of the tool through a Python class.

1. Import `MlFmuBuilder` and create an instance of it:

```python
from mlfmu.api import MlFmuBuilder
from pathlib import Path

builder = MlFmuBuilder(
    ml_model_file = Path("path/to/model.onnx")
    interface_file = Path("path/to/interface.json")
)
```

2. Call the same commands using the class:

- Run `build`

```python
builder.build()
```

- Run `codegen` and then `compile`

```python
builder.generate()

# Do something ...

builder.compile()
```
