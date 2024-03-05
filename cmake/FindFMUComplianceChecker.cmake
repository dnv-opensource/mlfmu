# Finds FMUComplianceChecker
# (https://github.com/modelica-tools/FMUComplianceChecker)
#
# The script sets the following variables:
#
#     FMUComplianceChecker_FOUND      - TRUE on success, FALSE on failure
#     FMUComplianceChecker_EXECUTABLE - The fmuCheck executable, if found.
#
# It also defines the following macro, which adds a new test (using add_test)
# that runs the FMU Compliance Checker on an FMU:
#
#     add_fmu_compliance_check(test_name fmu_path)
#

# Detect platform
if("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set(_os "darwin")
elseif("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    set(_os "linux")
elseif(WIN32)
    set(_os "win")
else()
    message(FATAL_ERROR "Unknown or unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
math(EXPR _wordSize 8*${CMAKE_SIZEOF_VOID_P})

find_program(
    FMUComplianceChecker_EXECUTABLE
    "fmuCheck.${_os}${_wordSize}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    FMUComplianceChecker
    DEFAULT_MSG
    FMUComplianceChecker_EXECUTABLE)

macro(add_fmu_compliance_check testName fmuPath)
    add_test(
        NAME "${testName}"
        COMMAND "${FMUComplianceChecker_EXECUTABLE}" "${fmuPath}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endmacro()
