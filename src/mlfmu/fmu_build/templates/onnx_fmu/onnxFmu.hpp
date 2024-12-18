#include <array>
#include <cppfmu_cs.hpp>
#include <onnxruntime_cxx_api.h>
#include <sstream>

// Include definitions specific for each FMU using the template
#include "model_definitions.h"

union FMIVariable {
    struct
    {
        cppfmu::FMIReal real;
    };
    struct
    {
        cppfmu::FMIInteger integer;
    };
    struct
    {
        cppfmu::FMIString string;
    };
    struct
    {
        cppfmu::FMIBoolean boolean;
    };
};

class OnnxFmu : public cppfmu::SlaveInstance
{
public:
    OnnxFmu(cppfmu::FMIString fmuResourceLocation);

    void formatOnnxPath(cppfmu::FMIString fmuResourceLocation);

    // New functions for the OnnxTemplate class
    void CreateSession();
    bool SetOnnxInputs();
    bool GetOnnxOutputs();
    bool SetOnnxStates();
    bool InitOnnxStates();
    bool RunOnnxModel(cppfmu::FMIReal currentCommunicationPoint, cppfmu::FMIReal dt);

    // Override functions from cppmu::SlaveInstance
    void Reset() override;
    void SetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIReal value[]) override;
    void GetReal(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIReal value[]) const override;
    void SetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIInteger value[]) override;
    void GetInteger(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIInteger value[]) const override;
    void SetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, const cppfmu::FMIBoolean value[]) override;
    void GetBoolean(const cppfmu::FMIValueReference vr[], std::size_t nvr, cppfmu::FMIBoolean value[]) const override;
    bool DoStep(cppfmu::FMIReal, cppfmu::FMIReal, cppfmu::FMIBoolean, cppfmu::FMIReal&) override;

private:
    std::array<FMIVariable, NUM_FMU_VARIABLES> fmuVariables_;

    Ort::Env env;
    Ort::RunOptions run_options;
    Ort::Session session_ {nullptr};

    // store path as wstring for Windows or as char * for Linux
#ifdef _WIN32
    std::wstring onnxPath_;
#else
    std::string onnxPath_;
#endif

    std::string inputName_ {ONNX_INPUT_NAME};
    std::array<int64_t, 2> inputShape_ {1, NUM_ONNX_INPUTS};
    std::array<float, NUM_ONNX_INPUTS> onnxInputs_ {};
    std::array<std::array<int, 2>, NUM_ONNX_FMU_INPUTS> onnxInputValueReferenceIndexPairs_ {ONNX_INPUT_VALUE_REFERENCES};

    std::string stateName_ {ONNX_STATE_NAME};
    std::array<int64_t, 2> stateShape_ {1, NUM_ONNX_STATES};
    std::array<float, NUM_ONNX_STATES> onnxStates_ {};
    std::array<int, NUM_ONNX_STATES_OUTPUTS> onnxStateOutputIndexes_ {ONNX_STATE_OUTPUT_INDEXES};

    std::string outputName_ {ONNX_OUTPUT_NAME};
    std::array<int64_t, 2> outputShape_ {1, NUM_ONNX_OUTPUTS};
    std::array<float, NUM_ONNX_OUTPUTS> onnxOutputs_ {};
    std::array<std::array<int, 2>, NUM_ONNX_FMU_OUTPUTS> onnxOutputValueReferenceIndexPairs_ {
        ONNX_OUTPUT_VALUE_REFERENCES};

    std::string timeInputName_ {ONNX_TIME_INPUT_NAME};
    std::array<int64_t, 2> timeInputShape_ {1, 2};
    std::array<float, 2> onnxTimeInput_ {0.0, 0.0};

    std::array<std::array<int, 2>, NUM_ONNX_STATE_INIT> onnxStateInitValueReferenceIndexPairs_ {
        ONNX_STATE_INIT_VALUE_REFERENCES};
    bool doStateInit_ = true;
};
