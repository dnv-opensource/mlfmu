# Changes to BASE_DIR and zips all files found there into OUTPUT_FILE.
if(NOT IS_ABSOLUTE "${BASE_DIR}")
    message(FATAL_ERROR "BASE_DIR does not contain an absolute path")
endif()
if(NOT IS_DIRECTORY "${BASE_DIR}")
    message(FATAL_ERROR "BASE_DIR does not refer to a directory")
endif()
if(NOT IS_ABSOLUTE "${OUTPUT_FILE}")
    message(FATAL_ERROR "OUTPUT_FILE does not contain an absolute path")
endif()

file(GLOB inputFiles RELATIVE "${BASE_DIR}" "${BASE_DIR}/*")
execute_process(
    COMMAND "${CMAKE_COMMAND}" "-E" "tar" "cf" "${OUTPUT_FILE}" "--format=zip" ${inputFiles}
    WORKING_DIRECTORY "${BASE_DIR}"
    RESULT_VARIABLE exitCode
)
if(NOT exitCode EQUAL 0)
    message(FATAL_ERROR "Zip command failed")
endif()
