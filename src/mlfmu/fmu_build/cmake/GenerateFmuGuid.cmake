# Generates a UUID to be used as the "guid" attribute of an FMU, and writes it
# to the modelDescription.xml file and to a header that can be imported in the
# FMU code.
#
# The script takes a template modelDescription.xml and header file as input,
# and substitutes a placeholder in each of them with the generated UUID before
# writing the output files.  The placeholder should be on the form "@FMU_UUID@"
# (the name is configurable, but the @ signs are not).
#
# The UUID will be generated as a hash of the input model description and,
# optionally, other input files (e.g. model source code files).
#
# Mandatory configuration variables:
#
#   INPUT_MODEL_DESCRIPTION
#       The model description XML file containing a placeholder in the "guid"
#       attribute.
#
#   INPUT_HEADER
#       A header file containing a placeholder for the UUID (typically in a
#       #define directive or constant declaration).
#
#   OUTPUT_MODEL_DESCRIPTION
#       The path to the output model description XML file.
#
#   OUTPUT_HEADER
#       The path to the output header file.
#
# Optional configuration variables:
#
#   ADDITIONAL_INPUT
#       Other input files that should be hashed into the UUID.
#
#   PLACEHOLDER_NAME
#       The name of the placeholder in the input model description and header.
#       The default value is "FMU_UUID".
#
cmake_minimum_required(VERSION 3.1)

if((NOT INPUT_MODEL_DESCRIPTION) OR (NOT INPUT_HEADER) OR (NOT OUTPUT_MODEL_DESCRIPTION) OR (NOT OUTPUT_HEADER))
    message(FATAL_ERROR "One or more mandatory configuration variables are not set")
endif()
if(NOT PLACEHOLDER_NAME)
    set(PLACEHOLDER_NAME "FMU_UUID")
endif()

macro(append_file_contents_without_newlines filePath outputVar)
    file(READ "${filePath}" contents)
    string(REGEX REPLACE "[\r\n]" "" contentsWithoutNewlines "${contents}")
    string(APPEND ${outputVar} "${contentsWithoutNewlines}")
endmacro()

# Read input model description file and any additional input files
set(text)
append_file_contents_without_newlines("${INPUT_MODEL_DESCRIPTION}" text)
foreach(f IN LISTS ADDITIONAL_INPUT)
    append_file_contents_without_newlines("${f}" text)
endforeach()

# Generate UUID
set(uuidNamespace "23d0734c-3983-4f65-a8f7-3bd5c5693546")
string(UUID ${PLACEHOLDER_NAME} NAMESPACE "${uuidNamespace}" NAME "${text}" TYPE "SHA1")

# Write output files
configure_file("${INPUT_MODEL_DESCRIPTION}" "${OUTPUT_MODEL_DESCRIPTION}" @ONLY)
configure_file("${INPUT_HEADER}" "${OUTPUT_HEADER}" @ONLY)
