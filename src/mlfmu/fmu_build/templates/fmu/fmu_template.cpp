#include "fmu-uuid.h"
#include "model_definitions.h"

#include <cppfmu_cs.hpp>
#include <onnxFmu.hpp>


/**
 * \class {FmuName}
 * \brief A class representing an {FmuName} FMU.
 *
 * This class is derived from the OnnxFmu class and provides functionality specific to the {FmuName} FMU.
 */
class {FmuName} : public OnnxFmu {{
    public :
        /**
         * \brief Constructs a new {FmuName} object.
         *
         * \param fmuResourceLocation The location of the resources of the FMU.
         */
        {FmuName}(cppfmu::FMIString fmuResourceLocation) : OnnxFmu(fmuResourceLocation) {{}}

    private :
        // Add private members and functions here
}};

/**
 * \brief Instantiate a `slave` instance of the FMU.
 *
 * \param instanceName The name of the FMU instance.
 * \param fmuGUID The GUID of the FMU.
 * \param fmuResourceLocation The location of the FMU resource.
 * \param mimeType The MIME type of the FMU.
 * \param timeout The timeout value for the instantiation process.
 * \param visible Flag indicating whether the FMU should be visible.
 * \param interactive Flag indicating whether the FMU should be interactive.
 * \param memory The memory to be used for the FMU instance.
 * \param logger The logger to be used for logging messages.
 * \returns A unique pointer to the instantiated slave instance.
 *
 * \throws std::runtime_error if the FMU GUID does not match.
 */
cppfmu::UniquePtr<cppfmu::SlaveInstance> CppfmuInstantiateSlave(
    cppfmu::FMIString /*instanceName*/, cppfmu::FMIString fmuGUID, cppfmu::FMIString fmuResourceLocation,
    cppfmu::FMIString /*mimeType*/, cppfmu::FMIReal /*timeout*/, cppfmu::FMIBoolean /*visible*/,
    cppfmu::FMIBoolean /*interactive*/, cppfmu::Memory memory, cppfmu::Logger /*logger*/)
{{
    if (std::strcmp(fmuGUID, FMU_UUID) != 0) {{
        throw std::runtime_error("FMU GUID mismatch");
    }}
    return cppfmu::AllocateUnique<{FmuName}>(memory, fmuResourceLocation);
}}
