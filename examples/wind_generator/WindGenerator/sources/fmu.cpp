#include "model_definitions.h"
#include "fmu-uuid.h"
#include <cppfmu_cs.hpp>
#include <onnxFmu.hpp>


class WindGenerator : public OnnxFmu{
public:
    WindGenerator(cppfmu::FMIString fmuResourceLocation) : OnnxFmu(fmuResourceLocation){}
private:
};


cppfmu::UniquePtr<cppfmu::SlaveInstance> CppfmuInstantiateSlave(
    cppfmu::FMIString /*instanceName*/, cppfmu::FMIString fmuGUID, cppfmu::FMIString fmuResourceLocation,
    cppfmu::FMIString /*mimeType*/, cppfmu::FMIReal /*timeout*/, cppfmu::FMIBoolean /*visible*/,
    cppfmu::FMIBoolean /*interactive*/, cppfmu::Memory memory, cppfmu::Logger /*logger*/)
{
    if (std::strcmp(fmuGUID, FMU_UUID) != 0) {
        throw std::runtime_error("FMU GUID mismatch");
    }
    return cppfmu::AllocateUnique<WindGenerator>(memory, fmuResourceLocation);
}
