#define CATCH_CONFIG_RUNNER

#include <catch2/catch_session.hpp>

#include "BenchConfig.h"


int main(int argc, char* argv[])
{
    GEMMBench::parseArguments(argc, argv);

    return Catch::Session().run(argc, argv);
}