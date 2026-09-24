# Determine target architecture for pre-installed Catch2
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv|rv64")
    set(_CATCH2_ARCH "riscv64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_CATCH2_ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_CATCH2_ARCH "x86_64")
else()
    set(_CATCH2_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(EXISTS "$ENV{HOME}/.local/catch2/${_CATCH2_ARCH}/lib/cmake/Catch2")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/catch2/${_CATCH2_ARCH}")
endif()

find_package(Catch2 3 REQUIRED)

add_executable(GemmBenchTests
    TestMain.cpp
    GemmUKernelTest.cpp
    AllocTest.cpp
    MacroKernelTest.cpp
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
