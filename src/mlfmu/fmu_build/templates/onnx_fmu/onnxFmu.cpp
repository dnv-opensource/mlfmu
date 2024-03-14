#include "onnxFmu.hpp"

OnnxFmu::OnnxFmu(cppfmu::FMIString fmuResourceLocation)
{
    onnxPath_ = formatOnnxPath(fmuResourceLocation);
    CreateSession();
    OnnxFmu::Reset();
}

std::wstring OnnxFmu::formatOnnxPath(cppfmu::FMIString fmuResourceLocation)
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
    return path;
}

void OnnxFmu::CreateSession()
{
    session_ = Ort::Session(env, onnxPath_.c_str(), Ort::SessionOptions {nullptr});
}

void OnnxFmu::Reset()
{
    doStateInit_ = true;
    return;
}

void OnnxFmu::SetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIReal value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        OnnxFmu::fmuVariables_[vr[i]].real = value[i];
    }
}

void OnnxFmu::GetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIReal value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].real;
    }
}

void OnnxFmu::SetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIInteger value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        fmuVariables_[vr[i]].integer = value[i];
    }
}

void OnnxFmu::GetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIInteger value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].integer;
    }
}

void OnnxFmu::SetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIBoolean value[])
{
    for (std::size_t i = 0; i < nvr; ++i) {
        fmuVariables_[vr[i]].boolean = value[i];
    }
}

void OnnxFmu::GetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIBoolean value[]) const
{
    for (std::size_t i = 0; i < nvr; ++i) {
        value[i] = fmuVariables_[vr[i]].boolean;
    }
}

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

bool OnnxFmu::SetOnnxStates()
{
    for (int index = 0; index < NUM_ONNX_STATES_OUTPUTS; index++) {
        onnxStates_[index] = onnxOutputs_[onnxStateOutputIndexes_[index]];
    }
    return true;
}

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
