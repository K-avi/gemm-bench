# gen_build_info.cmake
# Generates build_info.h with git, compiler, and build configuration info.

find_package(Git QUIET)

set(GEMMBENCH_GIT_COMMIT "unknown")
set(GEMMBENCH_GIT_DIRTY "-1")

if(GIT_FOUND AND EXISTS "${SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_RES
    )
    if(GIT_RES EQUAL 0 AND GIT_REV)
        set(GEMMBENCH_GIT_COMMIT "${GIT_REV}")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_STATUS_RES
    )
    if(GIT_STATUS_RES EQUAL 0)
        if(GIT_STATUS STREQUAL "")
            set(GEMMBENCH_GIT_DIRTY "0")
        else()
            set(GEMMBENCH_GIT_DIRTY "1")
        endif()
    endif()
endif()

# Flags
set(GEMMBENCH_CXX_FLAGS_COMBINED "${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    if(GEMMBENCH_CXX_FLAGS_COMBINED)
        set(GEMMBENCH_CXX_FLAGS_COMBINED "${GEMMBENCH_CXX_FLAGS_COMBINED} ${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        set(GEMMBENCH_CXX_FLAGS_COMBINED "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()
endif()
string(STRIP "${GEMMBENCH_CXX_FLAGS_COMBINED}" GEMMBENCH_CXX_FLAGS_COMBINED)

# Fastmath string
if(GEMMBENCH_ENABLE_FASTMATH)
    set(GEMMBENCH_FASTMATH_STR "ON")
else()
    set(GEMMBENCH_FASTMATH_STR "OFF")
endif()

# Explo string
if(GEMMBENCH_ENABLE_EXPLO)
    set(GEMMBENCH_EXPLO_STR "ON")
else()
    set(GEMMBENCH_EXPLO_STR "OFF")
endif()

# BLIS target
if(NOT GEMMBENCH_RESOLVED_BLIS_TARGET)
    set(GEMMBENCH_RESOLVED_BLIS_TARGET "none")
endif()

# Ensure output directory exists
file(MAKE_DIRECTORY "${BINARY_DIR}/generated")

# Configure to temporary file
configure_file(
    "${SOURCE_DIR}/cmake/build_info.h.in"
    "${BINARY_DIR}/generated/build_info.h.tmp"
    @ONLY
)

# Atomic update if changed
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${BINARY_DIR}/generated/build_info.h.tmp"
    "${BINARY_DIR}/generated/build_info.h"
)

file(REMOVE "${BINARY_DIR}/generated/build_info.h.tmp")
