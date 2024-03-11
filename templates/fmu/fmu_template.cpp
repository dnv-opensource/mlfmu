#include "fmu-uuid.h"
#include "model_definitions.h"

#include <cppfmu_cs.hpp>
#include <onnxFmu.hpp>


class
{
    FmuName
} : public OnnxFmu {{
    public :
        {FmuName}(cppfmu::FMIString fmuResourceLocation) : OnnxFmu(fmuResourceLocation) {{}} private :
}};


cppfmu::UniquePtr<cppfmu::SlaveInstance> CppfmuInstantiateSlave(
    cppfmu::FMIString /*instanceName*/, cppfmu::FMIString fmuGUID, cppfmu::FMIString fmuResourceLocation,
    cppfmu::FMIString /*mimeType*/, cppfmu::FMIReal /*timeout*/, cppfmu::FMIBoolean /*visible*/,
    cppfmu::FMIBoolean /*interactive*/, cppfmu::Memory memory, cppfmu::Logger /*logger*/)
{
    {
        if (std::strcmp(fmuGUID, FMU_UUID) != 0) {
            {
                throw std::runtime_error("FMU GUID mismatch");
            }
        }
        return cppfmu::AllocateUnique<{FmuName}>(memory, fmuResourceLocation);
    }
}
