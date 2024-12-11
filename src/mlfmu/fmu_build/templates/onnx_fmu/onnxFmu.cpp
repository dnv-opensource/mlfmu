#include "onnxFmu.hpp"

#include <iostream>

/**
 * \brief Constructs an instance of the OnnxFmu class.
 *
 * Constructs an instance of the OnnxFmu class.
 *
 * \param fmuResourceLocation The location of the FMU resource.
 */
OnnxFmu::OnnxFmu(cppfmu::FMIString fmuResourceLocation)
{
    formatOnnxPath(fmuResourceLocation);
    try {
        CreateSession();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << '\n';
    }
    OnnxFmu::Reset();
}


/**
 * \brief Formats the onnx path.
 *
 * Formats the onnx path by appending the ONNX_FILENAME to the given fmuResourceLocation.
 * If the path starts with "file:///", it removes the "file://" prefix.
 * This is directly stored to the class var onnxPath_.
 *
 * \param fmuResourceLocation The location of the FMU resource.
 */
void OnnxFmu::formatOnnxPath(cppfmu::FMIString fmuResourceLocation)
{
    // Creating complete path to onnx file
    std::wostringstream onnxPathStream;
    onnxPathStream << fmuResourceLocation;
    onnxPathStream << L"/";
    onnxPathStream << ONNX_FILENAME;

    // Remove file:// from the path if it is at the beginning
    std::wstring path = onnxPathStream.str();
    std::wstring startPath = path.substr(0, 8);
    std::wstring endPath = path.substr(8);
    if (startPath == L"file:///") {
        path = endPath;
    }
    // save to onnxPath_ (wstring for Windows, else string)
#ifdef _WIN32
    onnxPath_ = path;
#else
    onnxPath_ = std::string(path.begin(), path.end());
#endif
}

/**
 * \brief Creates a onnx runtime session for the model.
 *
 * This function creates a session to the ONNX model, using the specified ONNX model file.
 * This loads the weights of the model such that we can run predictions in the doStep function.
 *
 * \note The ONNX model file path must be set before calling this function.
 * \throws std::runtime_error if the session creation fails.
 */
void OnnxFmu::CreateSession()
{
    // Create the ONNX environment
    session_ = Ort::Session(env, onnxPath_.c_str(), Ort::SessionOptions {nullptr});
}

/**
 * \brief Resets the state of the OnnxFmu.
 *
 * This function is called to reset the state of the OnnxFmu.
 */
void OnnxFmu::Reset()
{
    doStateInit_ = true;
    return;
}

/**
 * Sets the real values of the specified FMI value references.
 *
 * \param vr An array of FMI value references.
 * \param nvr The number of FMI value references in the array.
 * \param value An array of FMI real values to be set.
 */
void OnnxFmu::SetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIReal value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        OnnxFmu::fmuVariables_[vr[i]].real = value[i];
    }
}


/**
 * \brief Retrieves the real values of the specified FMI value references.
 *
 * \param vr An array of FMI value references.
 * \param nvr The number of FMI value references in the array.
 * \param value An array to store the retrieved real values.
 */
void OnnxFmu::GetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIReal value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].real;
    }
}


/**
 * \brief Sets the integer values of the specified FMI value references.
 *
 * This function sets the integer values of the FMI value references specified in the
 * `vr` array to the corresponding values in the `value` array.
 *
 * \param vr An array of FMI value references.
 * \param nvr The number of FMI value references in the `vr` array.
 * \param value An array of FMI integer values.
 */
void OnnxFmu::SetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIInteger value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        fmuVariables_[vr[i]].integer = value[i];
    }
}


/**
 * \brief Retrieves the integer values of the specified FMI value references.
 *
 * This function retrieves the integer values of the FMI value references specified in the
 * `vr` array and stores them in the `value` array.
 *
 * \param vr An array of FMI value references.
 * \param nvr The number of FMI value references in the `vr` array.
 * \param value An array to store the retrieved integer values.
 */
void OnnxFmu::GetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIInteger value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].integer;
    }
}

/**
 * \brief Sets the boolean values for the specified FMI value references.
 *
 * This function sets the boolean values for the specified FMI value references in the
 * OnnxFmu object.
 *
 * \param vr An array of FMI value references.
 * \param nvr The number of FMI value references in the array.
 * \param value An array of FMI boolean values.
 */
void OnnxFmu::SetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIBoolean value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        fmuVariables_[vr[i]].boolean = value[i];
    }
}

/**
 * \brief Retrieves boolean values for the specified value references.
 *
 * This function retrieves boolean values for the specified value references from the
 * OnnxFmu object.
 *
 * \param vr An array of FMIValueReference representing the value references.
 * \param nvr The number of value references in the vr array.
 * \param value An array of FMIBoolean to store the retrieved boolean values.
 */
void OnnxFmu::GetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIBoolean value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].boolean;
    }
}

/**
 * \brief Sets the ONNX inputs for the ONNX FMU, matching FMU variables with inputs of ONNX model.
 *
 * This function matches the FMU variables with the inputs to the ONNX model.
 * It iterates over the ONNX input value-reference index pairs and assigns the corresponding FMU
 * variable's real value to the ONNX input.
 *
 * \returns `true` if the ONNX inputs are successfully set, `false` otherwise.
 */
bool OnnxFmu::SetOnnxInputs()
{
    for (int index = 0; index < NUM_ONNX_FMU_INPUTS; index++) {
        int inputIndex = onnxInputValueReferenceIndexPairs_[index][0];
        int valueReference = onnxInputValueReferenceIndexPairs_[index][1];
        FMIVariable var = fmuVariables_[valueReference];
        // TODO: Change to handle if the variable is not a real
        onnxInputs_[inputIndex] = var.real;
    }
    return true;
}

/**
 * \brief Sets the ONNX states.
 *
 * This function sets the ONNX states by assigning the corresponding ONNX outputs to the
 * ONNX states array.
 *
 * \returns `true` if the ONNX states are successfully set, `false` otherwise.
 */
bool OnnxFmu::SetOnnxStates()
{
    for (int index = 0; index < NUM_ONNX_STATES_OUTPUTS; index++) {
        onnxStates_[index] = onnxOutputs_[onnxStateOutputIndexes_[index]];
    }
    return true;
}

/**
 * \brief Retrieves the ONNX outputs and updates the corresponding FMU variables.
 *
 * This function iterates over the ONNX output value-reference index pairs and updates
 * the FMU variables with the corresponding ONNX outputs.
 * The function assumes that the FMU variables are of type `FMIVariable` and Real valued
 * and the ONNX outputs are stored in the `onnxOutputs_` array.
 *
 * \returns `true` if the ONNX outputs are successfully retrieved and FMU variables are
 * updated, `false` otherwise.
 */
bool OnnxFmu::GetOnnxOutputs()
{
    for (int index = 0; index < NUM_ONNX_FMU_OUTPUTS; index++) {
        int outputIndex = onnxOutputValueReferenceIndexPairs_[index][0];
        int valueReference = onnxOutputValueReferenceIndexPairs_[index][1];
        FMIVariable var = fmuVariables_[valueReference];
        // TODO: Change to handle if the variable is not a real
        var.real = onnxOutputs_[outputIndex];

        fmuVariables_[valueReference] = var;
    }
    return true;
}

/**
 * \brief Initializes the ONNX states of the ONNX model.
 *
 * This function initializes the ONNX states of the ONNX FMU by assigning the initial values
 * of the ONNX states from the corresponding variables in the FMU.
 * The function assumes that the FMU variables are of type `FMIVariable` and Real valued.
 *
 * \returns `true` if the ONNX states are successfully initialized, `false` otherwise.
 */
bool OnnxFmu::InitOnnxStates()
{
    for (int index = 0; index < NUM_ONNX_STATE_INIT; index++) {
        int stateIndex = onnxStateInitValueReferenceIndexPairs_[index][0];
        int valueReference = onnxStateInitValueReferenceIndexPairs_[index][1];
        FMIVariable var = fmuVariables_[valueReference];
        // TODO: Change to handle if the variable is not a real
        onnxStates_[stateIndex] = var.real;
    }
    return true;
}

/**
 * \brief Runs the ONNX model for the FMU.
 *
 * This function runs the ONNX model for the FMU using the provided current communication
 * point and time step.
 * It takes the input data, states (if any), and time inputs (if enabled) and calls Run
 * to produce the output data.
 *
 * \param currentCommunicationPoint The current communication point of the FMU.
 * \param dt The time step of the FMU.
 * \returns `true` if the ONNX model was successfully run, `false` otherwise.
 */
bool OnnxFmu::RunOnnxModel(cppfmu::FMIReal currentCommunicationPoint, cppfmu::FMIReal dt)
{
    try {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<Ort::Value> inputs;
        const char* inputNames[3] = {inputName_.c_str()};
        int numInputs = 1;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, onnxInputs_.data(), onnxInputs_.size(),
            inputShape_.data(), inputShape_.size()));

        if (NUM_ONNX_STATES > 0) {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, onnxStates_.data(), onnxStates_.size(),
                stateShape_.data(), stateShape_.size()));
            inputNames[1] = stateName_.c_str();
            numInputs++;
        }

        if (ONNX_USE_TIME_INPUT) {
            onnxTimeInput_[0] = currentCommunicationPoint;
            onnxTimeInput_[1] = dt;
            inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, onnxTimeInput_.data(),
                onnxTimeInput_.size(), timeInputShape_.data(),
                timeInputShape_.size()));
            inputNames[2] = timeInputName_.c_str();
            numInputs++;
        }

        const char* output_names[] = {outputName_.c_str()};
        Ort::Value outputs = Ort::Value::CreateTensor<float>(memoryInfo, onnxOutputs_.data(), onnxOutputs_.size(),
            outputShape_.data(), outputShape_.size());

        session_.Run(run_options, inputNames, inputs.data(), numInputs, output_names, &outputs, 1);
    } catch (const std::exception& /*e*/) {
        return false;
    }
    return true;
}

/**
 * \brief Performs a step in the OnnxFmu simulation.
 *
 * This function is called by the FMU framework to perform a step in the simulation.
 * It initializes the ONNX states if necessary, sets the ONNX inputs, runs the ONNX model,
 * gets the ONNX outputs, and sets the ONNX states.
 *
 * \param currentCommunicationPoint The current communication point in the simulation.
 * \param dt The time step size for the simulation.
 * \param newStep A boolean indicating whether a new step has started.
 * \param endOfStep The end of the current step in the simulation.
 * \returns `true` if the step was successful, `false` otherwise.
 */
bool OnnxFmu::DoStep(cppfmu::FMIReal currentCommunicationPoint, cppfmu::FMIReal dt, cppfmu::FMIBoolean /*newStep*/,
    cppfmu::FMIReal& /*endOfStep*/)
{
    if (doStateInit_) {
        bool initOnnxStates = InitOnnxStates();
        if (!initOnnxStates) {
            return false;
        }
        doStateInit_ = false;
    }
    bool setOnnxSuccessful = SetOnnxInputs();
    if (!setOnnxSuccessful) {
        return false;
    }
    bool runOnnxSuccessful = RunOnnxModel(currentCommunicationPoint, dt);
    if (!runOnnxSuccessful) {
        return false;
    }
    bool getOnnxSuccessful = GetOnnxOutputs();
    if (!getOnnxSuccessful) {
        return false;
    }
    bool setOnnxStateSuccessful = SetOnnxStates();
    if (!setOnnxStateSuccessful) {
        return false;
    }
    return true;
}
