# Look for pre-installed Catch2 in ~/.local/catch2/<arch> if present
if(EXISTS "$ENV{HOME}/.local/catch2/${CMAKE_SYSTEM_PROCESSOR}")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/catch2/${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    include(FetchContent)
    message(STATUS "Catch2 v3 not found locally. Fetching from upstream...")
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(GemmBenchTests
    TestMain.cpp
    GemmUKernelTest.cpp
    AllocTest.cpp
)

target_compile_options(GemmBenchTests
    PRIVATE
        -O2
)

target_link_libraries(GemmBenchTests
    PRIVATE
        gemmbench
        Catch2::Catch2
)

include(CTest)
include(Catch)

catch_discover_tests(GemmBenchTests)
